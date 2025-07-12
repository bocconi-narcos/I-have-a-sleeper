import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights
from src.models.base.transformer_blocks import PreNorm, FeedForward, Attention, Transformer


class ARC_TransitionModel(nn.Module):
    """
    Transformer-based predictor for the next latent state (x_{t+1}).
    
    Inputs:
        - encoded_state: Tensor of shape (batch_size, state_dim)
        - encoded_action: Tensor of shape (batch_size, action_dim)
    Output:
        - predicted_next_latent: Tensor of shape (batch_size, latent_dim)
    """
    
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int = None,
                 transformer_depth: int = 2, transformer_heads: int = 2, 
                 transformer_dim_head: int = 64, transformer_mlp_dim: int = 128, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim or state_dim  # output dim

        # Project both inputs to the same dimension if needed
        self.state_proj = nn.Linear(state_dim, self.latent_dim) if state_dim != self.latent_dim else nn.Identity()
        self.action_proj = nn.Linear(action_dim, self.latent_dim)

        # Positional encoding for sequence of length 2 (state + action)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, self.latent_dim))

        self.transformer = Transformer(
            dim=self.latent_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )
        
        self.output_proj = nn.Linear(self.latent_dim, self.latent_dim)
        
        # Apply weight initialization
        self.apply(initialize_weights)
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Print model info
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[TransitionModel] Number of parameters: {num_params}")

    def forward(self, encoded_state: torch.Tensor, encoded_action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state embedding given current state and action embeddings.
        
        Args:
            encoded_state (torch.Tensor): Current state embedding of shape (B, state_dim)
            encoded_action (torch.Tensor): Action embedding of shape (B, action_dim)
            
        Returns:
            torch.Tensor: Predicted next state embedding of shape (B, latent_dim)
        """
        # Validate input dimensions
        if encoded_state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state embedding dimension {self.state_dim}, "
                           f"got {encoded_state.shape[-1]}")
        if encoded_action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action embedding dimension {self.action_dim}, "
                           f"got {encoded_action.shape[-1]}")
        if encoded_state.shape[0] != encoded_action.shape[0]:
            raise ValueError(f"Batch size mismatch: state {encoded_state.shape[0]}, "
                           f"action {encoded_action.shape[0]}")
        
        # Project to common dimension
        state_proj = self.state_proj(encoded_state)   # (B, latent_dim)
        action_proj = self.action_proj(encoded_action) # (B, latent_dim)
        
        # Stack as sequence: [state, action]
        x = torch.stack([state_proj, action_proj], dim=1)  # (B, 2, latent_dim)
        
        # Add positional encoding
        x = x + self.pos_embed  # (B, 2, latent_dim)
        
        # Pass through transformer
        x = self.transformer(x)  # (B, 2, latent_dim)
        
        # Use the first token (state position) as the output
        predicted_next_latent = self.output_proj(x[:, 0])  # (B, latent_dim)
        
        return predicted_next_latent
