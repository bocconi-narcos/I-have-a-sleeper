import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from src.models.ARC_specific.ARC_state_encoder import ARC_StateEncoder
from src.models.ARC_specific.ARC_action_encoder import ARC_ActionEncoder
from src.models.ARC_specific.ARC_transition_model import ARC_TransitionModel
from src.data import ReplayBufferDataset
from tqdm import tqdm
import torch.nn.functional as F

# --- EMA Utility ---
def update_ema(target_model, source_model, decay=0.995):
    """Update target model parameters using exponential moving average."""
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

# --- Config Loader ---
def load_config(config_path="configs/train_JEPA_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- Validation Function ---
def evaluate_model(state_encoder, target_encoder, action_encoder, transition_model, 
                   dataloader, device, criterion):
    """Evaluate the model on validation data."""
    state_encoder.eval()
    target_encoder.eval()
    action_encoder.eval()
    transition_model.eval()
    
    total_loss = 0
    total_cosine_sim = 0
    num_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action = batch['action'].to(device)
            
            # Additional parameters for ARC_StateEncoder
            shape_h = batch['shape_h'].to(device)
            shape_w = batch['shape_w'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            
            # Next state parameters
            shape_h_next = batch['shape_h_next'].to(device)
            shape_w_next = batch['shape_w_next'].to(device)
            most_present_color_next = batch['most_present_color_next'].to(device)
            least_present_color_next = batch['least_present_color_next'].to(device)
            num_colors_grid_next = batch['num_colors_grid_next'].to(device)
            
            # Encode current state with online encoder
            current_state_encoded = state_encoder(
                state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid
            )
            
            # Encode next state with target encoder (EMA)
            next_state_encoded = target_encoder(
                next_state, shape_h_next, shape_w_next, most_present_color_next, least_present_color_next, num_colors_grid_next
            )
            
            # Encode action
            action_encoded = action_encoder(action)
            
            # Predict next state encoding
            predicted_next_state = transition_model(current_state_encoded, action_encoded)
            
            # Compute loss between predicted and actual next state encodings
            loss = criterion(predicted_next_state, next_state_encoded)
            
            # Compute cosine similarity for monitoring
            predicted_norm = F.normalize(predicted_next_state, p=2, dim=-1)
            target_norm = F.normalize(next_state_encoded, p=2, dim=-1)
            cosine_sim = (predicted_norm * target_norm).sum(dim=-1).mean()
            
            batch_size = state.size(0)
            total_loss += loss.item() * batch_size
            total_cosine_sim += cosine_sim.item() * batch_size
            num_samples += batch_size
    
    avg_loss = total_loss / num_samples
    avg_cosine_sim = total_cosine_sim / num_samples
    
    return avg_loss, avg_cosine_sim

# --- Main Training Function ---
def train_jepa_world_model():
    """
    Main training loop for JEPA-style world model.
    
    The model consists of:
    1. State encoder: encodes current state
    2. Target encoder: EMA of state encoder, encodes next state
    3. Action encoder: encodes action
    4. Transition model: predicts next state embedding from current state + action
    
    Loss is computed between predicted next state embedding and actual next state embedding.
    """
    config = load_config()
    
    # Extract relevant config sections
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    
    # Training parameters
    batch_size = training_config['batch_size']
    num_epochs = training_config['num_epochs']
    learning_rate = training_config['learning_rate']
    num_workers = data_config.get('num_workers', 4)
    log_interval = training_config.get('log_interval', 10)
    
    # EMA parameters
    ema_decay = training_config.get('ema_decay', 0.995)
    
    # Device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print('Using device: MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: CUDA')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')
    
    # Create dataset
    dataset = ReplayBufferDataset(
        buffer_path=data_config['buffer_path'],
        state_shape=tuple(data_config['state_shape']),
        mode=data_config.get('mode', 'color_only'),
        num_samples=data_config.get('num_samples', None)
    )
    
    # Split dataset
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Initialize models
    state_encoder = ARC_StateEncoder(
        image_size=model_config['image_size'],
        input_channels=model_config['input_channels'],
        latent_dim=model_config['latent_dim_state'],
        encoder_params=model_config.get('encoder_params', {})
    ).to(device)
    
    # Target encoder (EMA copy of state encoder)
    target_encoder = copy.deepcopy(state_encoder)
    target_encoder.eval()
    for param in target_encoder.parameters():
        param.requires_grad = False
    
    # Action encoder (g) - Use embedding approach for JSAE
    action_encoder = ARC_ActionEncoder(
        num_actions=model_config['num_actions'],
        embedding_dim=model_config['latent_dim_action'],
        encoder_type='embedding'  # Force embedding approach for JSAE
    ).to(device)
    
    transition_model = ARC_TransitionModel(
        state_dim=model_config['latent_dim_state'],
        action_dim=model_config['latent_dim_action'],
        latent_dim=model_config['latent_dim_state'],
        transformer_depth=model_config.get('transition_depth', 2),
        transformer_heads=model_config.get('transition_heads', 4),
        transformer_dim_head=model_config.get('transition_dim_head', 64),
        transformer_mlp_dim=model_config.get('transition_mlp_dim', 256),
        dropout=model_config.get('dropout', 0.0)
    ).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in state_encoder.parameters()) + \
                   sum(p.numel() for p in action_encoder.parameters()) + \
                   sum(p.numel() for p in transition_model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    
    # Log JSAE configuration
    print("Using JSAE approach:")
    print(f"  Action encoder: Embedding table ({model_config['num_actions']} actions -> {model_config['latent_dim_action']}D)")
    print(f"  Note: JEPA uses no action decoder - loss computed on next state embeddings")
    
    # Initialize optimizer (only for trainable models, not target encoder)
    optimizer = optim.AdamW(
        list(state_encoder.parameters()) + 
        list(action_encoder.parameters()) + 
        list(transition_model.parameters()),
        lr=learning_rate,
        weight_decay=training_config.get('weight_decay', 1e-4)
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Initialize wandb
    wandb.init(project="jepa-world-model", config=config)
    
    # Training loop
    best_val_loss = float('inf')
    patience = training_config.get('patience', 10)
    epochs_no_improve = 0
    save_path = 'outputs/world_model_training/best_jepa_model.pt'
    
    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        # Training phase
        state_encoder.train()
        action_encoder.train()
        transition_model.train()
        
        total_train_loss = 0
        total_train_cosine = 0
        num_train_samples = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action = batch['action'].to(device)
            
            # Additional parameters for ARC_StateEncoder
            shape_h = batch['shape_h'].to(device)
            shape_w = batch['shape_w'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            
            # Next state parameters
            shape_h_next = batch['shape_h_next'].to(device)
            shape_w_next = batch['shape_w_next'].to(device)
            most_present_color_next = batch['most_present_color_next'].to(device)
            least_present_color_next = batch['least_present_color_next'].to(device)
            num_colors_grid_next = batch['num_colors_grid_next'].to(device)
            
            # Forward pass
            current_state_encoded = state_encoder(
                state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid
            )
            next_state_encoded = target_encoder(
                next_state, shape_h_next, shape_w_next, most_present_color_next, least_present_color_next, num_colors_grid_next
            )
            action_encoded = action_encoder(action)
            
            # Predict next state encoding
            predicted_next_state = transition_model(current_state_encoded, action_encoded)
            
            # Compute loss
            loss = criterion(predicted_next_state, next_state_encoded)
            
            # Compute cosine similarity for monitoring
            predicted_norm = F.normalize(predicted_next_state, p=2, dim=-1)
            target_norm = F.normalize(next_state_encoded, p=2, dim=-1)
            cosine_sim = (predicted_norm * target_norm).sum(dim=-1).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) + 
                list(action_encoder.parameters()) + 
                list(transition_model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            # Update target encoder with EMA
            update_ema(target_encoder, state_encoder, decay=ema_decay)
            
            # Track metrics
            batch_size = state.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_cosine += cosine_sim.item() * batch_size
            num_train_samples += batch_size
            
            # Log batch metrics
            if (i + 1) % log_interval == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_cosine_sim": cosine_sim.item(),
                    "epoch": epoch + 1,
                    "batch": i + 1
                })
        
        # Calculate average training metrics
        avg_train_loss = total_train_loss / num_train_samples
        avg_train_cosine = total_train_cosine / num_train_samples
        
        # Validation phase
        val_loss, val_cosine_sim = evaluate_model(
            state_encoder, target_encoder, action_encoder, transition_model,
            val_loader, device, criterion
        )
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Cosine: {avg_train_cosine:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Cosine: {val_cosine_sim:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_cosine_sim": avg_train_cosine,
            "val_loss": val_loss,
            "val_cosine_sim": val_cosine_sim
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'state_encoder': state_encoder.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'action_encoder': action_encoder.state_dict(),
                'transition_model': transition_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement for {patience} epochs.")
            break
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    train_jepa_world_model() 