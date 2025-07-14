#!/usr/bin/env python3
"""
Combined Generative and JEPA World Model Training Script

This script implements a training pipeline that simultaneously trains:
1. Generative World Model - Supervised learning with reconstruction losses
2. JEPA World Model - Self-supervised learning with prediction losses

Usage:
    python -m src.scripts.train_combined_wm --config configs/train_combined_config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import yaml
import argparse
import os
import time
import copy
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import numpy as np
from tqdm import tqdm
import wandb

# Import the models
from src.models.ARC_specific.ARC_state_encoder import ARC_StateEncoder
from src.models.ARC_specific.ARC_state_decoder import ARC_StateDecoder
from src.models.ARC_specific.ARC_action_encoder import ARC_ActionEncoder
from src.models.ARC_specific.ARC_transition_model import ARC_TransitionModel
from src.models.ARC_specific.ARC_action_decoder import ARC_ActionDecoder
from src.data.replay_buffer_dataset import ReplayBufferDataset


def update_ema(target_model, source_model, decay=0.995):
    """Update target model parameters using exponential moving average."""
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)


class CombinedWorldModelTrainer:
    """
    Trainer class for simultaneous generative and JEPA world model training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Device selection
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
            print('Using device: MPS (Apple Silicon GPU)')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print('Using device: CUDA')
        else:
            self.device = torch.device('cpu')
            print('Using device: CPU')
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.setup_models()
        
        # Initialize optimizers
        self.setup_optimizers()
        
        # Initialize datasets
        self.setup_datasets()
        
        # Training metrics for both models
        self.train_metrics = {
            'generative': {'transition_loss': [], 'action_loss': [], 'total_loss': []},
            'jepa': {'prediction_loss': [], 'cosine_sim': []}
        }
        self.val_metrics = {
            'generative': {'transition_loss': [], 'action_loss': [], 'total_loss': []},
            'jepa': {'prediction_loss': [], 'cosine_sim': []}
        }
        
        # EMA decay for JEPA
        self.ema_decay = config['training'].get('ema_decay', 0.995)
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'combined_training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Combined world model training started. Output directory: {self.output_dir}")
        
    def setup_models(self):
        """Initialize all model components for both generative and JEPA training."""
        model_config = self.config['model']
        
        # === GENERATIVE MODEL COMPONENTS ===
        
        # Generative state encoder
        self.gen_state_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        # Generative state decoder
        self.gen_state_decoder = ARC_StateDecoder(
            latent_dim=model_config['latent_dim_state'],
            decoder_params=model_config.get('decoder_params', {})
        ).to(self.device)
        
        # Generative action encoder
        self.gen_action_encoder = ARC_ActionEncoder(
            num_actions=model_config['num_actions'],
            embedding_dim=model_config['latent_dim_action'],
            encoder_type='embedding'
        ).to(self.device)
        
        # Generative action decoder
        self.gen_action_decoder = ARC_ActionDecoder(
            embedding_dim=model_config['latent_dim_action'],
            num_actions=model_config['num_actions'],
            hidden_dims=model_config.get('action_decoder_hidden_dims', [512, 256]),
            dropout=model_config.get('dropout', 0.1)
        ).to(self.device)
        
        # Generative transition model
        self.gen_transition_model = ARC_TransitionModel(
            state_dim=model_config['latent_dim_state'],
            action_dim=model_config['latent_dim_action'],
            latent_dim=model_config['latent_dim_state'],
            transformer_depth=model_config.get('transition_depth', 2),
            transformer_heads=model_config.get('transition_heads', 4),
            transformer_dim_head=model_config.get('transition_dim_head', 64),
            transformer_mlp_dim=model_config.get('transition_mlp_dim', 256),
            dropout=model_config.get('dropout', 0.0)
        ).to(self.device)
        
        # === JEPA MODEL COMPONENTS ===
        
        # JEPA state encoder (online)
        self.jepa_state_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        # JEPA target encoder (EMA copy)
        self.jepa_target_encoder = copy.deepcopy(self.jepa_state_encoder)
        self.jepa_target_encoder.eval()
        for param in self.jepa_target_encoder.parameters():
            param.requires_grad = False
        
        # JEPA action encoder
        self.jepa_action_encoder = ARC_ActionEncoder(
            num_actions=model_config['num_actions'],
            embedding_dim=model_config['latent_dim_action'],
            encoder_type='embedding'
        ).to(self.device)
        
        # JEPA transition model
        self.jepa_transition_model = ARC_TransitionModel(
            state_dim=model_config['latent_dim_state'],
            action_dim=model_config['latent_dim_action'],
            latent_dim=model_config['latent_dim_state'],
            transformer_depth=model_config.get('transition_depth', 2),
            transformer_heads=model_config.get('transition_heads', 4),
            transformer_dim_head=model_config.get('transition_dim_head', 64),
            transformer_mlp_dim=model_config.get('transition_mlp_dim', 256),
            dropout=model_config.get('dropout', 0.0)
        ).to(self.device)
        
        # Print model info
        gen_params = sum(p.numel() for p in self.gen_state_encoder.parameters()) + \
                    sum(p.numel() for p in self.gen_state_decoder.parameters()) + \
                    sum(p.numel() for p in self.gen_action_encoder.parameters()) + \
                    sum(p.numel() for p in self.gen_action_decoder.parameters()) + \
                    sum(p.numel() for p in self.gen_transition_model.parameters())
        
        jepa_params = sum(p.numel() for p in self.jepa_state_encoder.parameters()) + \
                     sum(p.numel() for p in self.jepa_action_encoder.parameters()) + \
                     sum(p.numel() for p in self.jepa_transition_model.parameters())
        
        self.logger.info(f"Generative model trainable parameters: {gen_params:,}")
        self.logger.info(f"JEPA model trainable parameters: {jepa_params:,}")
        self.logger.info(f"Total trainable parameters: {gen_params + jepa_params:,}")
    
    def setup_optimizers(self):
        """Initialize optimizers for both models."""
        training_config = self.config['training']
        
        # Generative model optimizer
        gen_params = list(self.gen_state_encoder.parameters()) + \
                    list(self.gen_state_decoder.parameters()) + \
                    list(self.gen_action_encoder.parameters()) + \
                    list(self.gen_action_decoder.parameters()) + \
                    list(self.gen_transition_model.parameters())
        
        self.gen_optimizer = torch.optim.AdamW(
            gen_params,
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # JEPA model optimizer
        jepa_params = list(self.jepa_state_encoder.parameters()) + \
                     list(self.jepa_action_encoder.parameters()) + \
                     list(self.jepa_transition_model.parameters())
        
        self.jepa_optimizer = torch.optim.AdamW(
            jepa_params,
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # Learning rate schedulers
        if training_config.get('use_scheduler', False):
            self.gen_scheduler = torch.optim.lr_scheduler.StepLR(
                self.gen_optimizer,
                step_size=training_config.get('scheduler_step_size', 10),
                gamma=training_config.get('scheduler_gamma', 0.1)
            )
            self.jepa_scheduler = torch.optim.lr_scheduler.StepLR(
                self.jepa_optimizer,
                step_size=training_config.get('scheduler_step_size', 10),
                gamma=training_config.get('scheduler_gamma', 0.1)
            )
        else:
            self.gen_scheduler = None
            self.jepa_scheduler = None
    
    def setup_datasets(self):
        """Initialize datasets and data loaders."""
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Create dataset
        dataset = ReplayBufferDataset(
            buffer_path=data_config['buffer_path'],
            state_shape=tuple(data_config['state_shape']),
            mode=data_config.get('mode', 'color_only'),
            num_samples=data_config.get('num_samples', None)
        )
        
        self.logger.info(f"Dataset loaded with {len(dataset)} samples")
        
        # Split dataset
        train_size = int(data_config['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        batch_size = training_config['batch_size']
        num_workers = data_config.get('num_workers', 4)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        self.logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
    
    def compute_generative_losses(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute generative model losses (transition and action reconstruction).
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            Tuple of (total_loss, transition_loss, action_loss)
        """
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        
        # Encode current state
        x_t = self.gen_state_encoder(
            batch['state'],
            batch['shape_h'],
            batch['shape_w'],
            batch['most_present_color'],
            batch['least_present_color'],
            batch['num_colors_grid']
        )
        
        # Encode actions
        e_t = self.gen_action_encoder(batch['action'])
        
        # Predict next state embedding using transition model
        x_next_pred = self.gen_transition_model(x_t, e_t)
        
        # Decode predicted next state to get logits
        decoded_next_state = self.gen_state_decoder(x_next_pred)
        
        # Compute transition loss as cross-entropy between decoded logits and actual next state
        transition_loss = self._compute_transition_loss(decoded_next_state, batch)
        
        # Compute action reconstruction loss using JSAE approach
        action_logits = self.gen_action_decoder.get_action_logits(e_t, temperature=1.0)
        action_loss = F.cross_entropy(action_logits, batch['action'])
        
        # Combine losses
        alpha = self.config['training']['action_loss_weight']
        total_loss = transition_loss + alpha * action_loss
        
        return total_loss, transition_loss, action_loss
    
    def _compute_transition_loss(self, decoded_state: Dict[str, torch.Tensor], 
                               batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute transition loss between decoded next state and actual next state.
        
        Args:
            decoded_state: Dictionary containing decoded state components
            batch: Batch containing actual next state data
            
        Returns:
            Combined transition loss
        """
        losses = []
        
        # Grid reconstruction loss
        if 'grid_logits' in decoded_state:
            grid_loss = F.cross_entropy(
                decoded_state['grid_logits'].view(-1, decoded_state['grid_logits'].size(-1)),
                batch['next_state'].view(-1).long()
            )
            losses.append(grid_loss)
        
        # Shape losses
        if 'shape_h_logits' in decoded_state:
            shape_h_loss = F.cross_entropy(decoded_state['shape_h_logits'], batch['shape_h_next'].long())
            losses.append(shape_h_loss * 0.1)  # Weight shape losses less
            
        if 'shape_w_logits' in decoded_state:
            shape_w_loss = F.cross_entropy(decoded_state['shape_w_logits'], batch['shape_w_next'].long())
            losses.append(shape_w_loss * 0.1)
        
        # Color losses
        if 'most_present_color_logits' in decoded_state:
            most_color_loss = F.cross_entropy(decoded_state['most_present_color_logits'], batch['most_present_color_next'].long())
            losses.append(most_color_loss * 0.1)
            
        if 'least_present_color_logits' in decoded_state:
            least_color_loss = F.cross_entropy(decoded_state['least_present_color_logits'], batch['least_present_color_next'].long())
            losses.append(least_color_loss * 0.1)
            
        if 'num_colors_grid_logits' in decoded_state:
            num_colors_loss = F.cross_entropy(decoded_state['num_colors_grid_logits'], batch['num_colors_grid_next'].long())
            losses.append(num_colors_loss * 0.1)
        
        return sum(losses) if losses else torch.tensor(0.0, device=self.device)
    
    def compute_jepa_losses(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute JEPA model losses (prediction loss between embeddings).
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            Tuple of (prediction_loss, cosine_similarity)
        """
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        
        # Encode current state with online encoder
        current_state_encoded = self.jepa_state_encoder(
            batch['state'],
            batch['shape_h'],
            batch['shape_w'],
            batch['most_present_color'],
            batch['least_present_color'],
            batch['num_colors_grid']
        )
        
        # Encode next state with target encoder (EMA)
        with torch.no_grad():
            next_state_encoded = self.jepa_target_encoder(
                batch['next_state'],
                batch['shape_h_next'],
                batch['shape_w_next'],
                batch['most_present_color_next'],
                batch['least_present_color_next'],
                batch['num_colors_grid_next']
            )
        
        # Encode action
        action_encoded = self.jepa_action_encoder(batch['action'])
        
        # Predict next state encoding
        predicted_next_state = self.jepa_transition_model(current_state_encoded, action_encoded)
        
        # Compute prediction loss
        prediction_loss = F.mse_loss(predicted_next_state, next_state_encoded)
        
        # Compute cosine similarity for monitoring
        with torch.no_grad():
            predicted_norm = F.normalize(predicted_next_state, p=2, dim=-1)
            target_norm = F.normalize(next_state_encoded, p=2, dim=-1)
            cosine_sim = (predicted_norm * target_norm).sum(dim=-1).mean()
        
        return prediction_loss, cosine_sim
    
    def train_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """Train both models for one epoch."""
        # Set models to training mode
        self.gen_state_encoder.train()
        self.gen_state_decoder.train()
        self.gen_action_encoder.train()
        self.gen_action_decoder.train()
        self.gen_transition_model.train()
        
        self.jepa_state_encoder.train()
        self.jepa_action_encoder.train()
        self.jepa_transition_model.train()
        
        # Track metrics for both models
        gen_losses = {'transition_loss': [], 'action_loss': [], 'total_loss': []}
        jepa_losses = {'prediction_loss': [], 'cosine_sim': []}
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # === GENERATIVE MODEL TRAINING ===
            self.gen_optimizer.zero_grad()
            gen_total_loss, gen_transition_loss, gen_action_loss = self.compute_generative_losses(batch)
            gen_total_loss.backward()
            
            # Gradient clipping for generative model
            if self.config['training'].get('grad_clip', 0) > 0:
                gen_params = list(self.gen_state_encoder.parameters()) + \
                           list(self.gen_state_decoder.parameters()) + \
                           list(self.gen_action_encoder.parameters()) + \
                           list(self.gen_action_decoder.parameters()) + \
                           list(self.gen_transition_model.parameters())
                torch.nn.utils.clip_grad_norm_(gen_params, self.config['training']['grad_clip'])
            
            self.gen_optimizer.step()
            
            # === JEPA MODEL TRAINING ===
            self.jepa_optimizer.zero_grad()
            jepa_prediction_loss, jepa_cosine_sim = self.compute_jepa_losses(batch)
            jepa_prediction_loss.backward()
            
            # Gradient clipping for JEPA model
            if self.config['training'].get('grad_clip', 0) > 0:
                jepa_params = list(self.jepa_state_encoder.parameters()) + \
                            list(self.jepa_action_encoder.parameters()) + \
                            list(self.jepa_transition_model.parameters())
                torch.nn.utils.clip_grad_norm_(jepa_params, self.config['training']['grad_clip'])
            
            self.jepa_optimizer.step()
            
            # Update JEPA target encoder with EMA
            update_ema(self.jepa_target_encoder, self.jepa_state_encoder, decay=self.ema_decay)
            
            # Store losses
            gen_losses['total_loss'].append(gen_total_loss.item())
            gen_losses['transition_loss'].append(gen_transition_loss.item())
            gen_losses['action_loss'].append(gen_action_loss.item())
            
            jepa_losses['prediction_loss'].append(jepa_prediction_loss.item())
            jepa_losses['cosine_sim'].append(jepa_cosine_sim.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Gen Total': f"{gen_total_loss.item():.4f}",
                'Gen Trans': f"{gen_transition_loss.item():.4f}",
                'Gen Action': f"{gen_action_loss.item():.4f}",
                'JEPA Pred': f"{jepa_prediction_loss.item():.4f}",
                'JEPA Cos': f"{jepa_cosine_sim.item():.4f}"
            })
            
            # Log progress
            if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                    f'Gen Total: {gen_total_loss.item():.4f}, '
                    f'Gen Trans: {gen_transition_loss.item():.4f}, '
                    f'Gen Action: {gen_action_loss.item():.4f}, '
                    f'JEPA Pred: {jepa_prediction_loss.item():.4f}, '
                    f'JEPA Cos: {jepa_cosine_sim.item():.4f}'
                )
        
        # Compute average losses
        avg_metrics = {
            'generative': {key: np.mean(values) for key, values in gen_losses.items()},
            'jepa': {key: np.mean(values) for key, values in jepa_losses.items()}
        }
        
        # Update schedulers
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.jepa_scheduler is not None:
            self.jepa_scheduler.step()
            
        return avg_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """Validate both models for one epoch."""
        # Set models to evaluation mode
        self.gen_state_encoder.eval()
        self.gen_state_decoder.eval()
        self.gen_action_encoder.eval()
        self.gen_action_decoder.eval()
        self.gen_transition_model.eval()
        
        self.jepa_state_encoder.eval()
        self.jepa_action_encoder.eval()
        self.jepa_transition_model.eval()
        
        # Track metrics for both models
        gen_losses = {'transition_loss': [], 'action_loss': [], 'total_loss': []}
        jepa_losses = {'prediction_loss': [], 'cosine_sim': []}
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Generative model validation
                gen_total_loss, gen_transition_loss, gen_action_loss = self.compute_generative_losses(batch)
                gen_losses['total_loss'].append(gen_total_loss.item())
                gen_losses['transition_loss'].append(gen_transition_loss.item())
                gen_losses['action_loss'].append(gen_action_loss.item())
                
                # JEPA model validation
                jepa_prediction_loss, jepa_cosine_sim = self.compute_jepa_losses(batch)
                jepa_losses['prediction_loss'].append(jepa_prediction_loss.item())
                jepa_losses['cosine_sim'].append(jepa_cosine_sim.item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen Total': f"{gen_total_loss.item():.4f}",
                    'JEPA Pred': f"{jepa_prediction_loss.item():.4f}",
                    'JEPA Cos': f"{jepa_cosine_sim.item():.4f}"
                })
        
        # Compute average losses
        avg_metrics = {
            'generative': {key: np.mean(values) for key, values in gen_losses.items()},
            'jepa': {key: np.mean(values) for key, values in jepa_losses.items()}
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Dict[str, float]], 
                       is_best_gen: bool = False, is_best_jepa: bool = False):
        """Save model checkpoints for both models."""
        
        # Save generative model checkpoint
        gen_checkpoint = {
            'epoch': epoch,
            'gen_state_encoder_state_dict': self.gen_state_encoder.state_dict(),
            'gen_state_decoder_state_dict': self.gen_state_decoder.state_dict(),
            'gen_action_encoder_state_dict': self.gen_action_encoder.state_dict(),
            'gen_action_decoder_state_dict': self.gen_action_decoder.state_dict(),
            'gen_transition_model_state_dict': self.gen_transition_model.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'gen_metrics': metrics['generative'],
            'config': self.config
        }
        
        if self.gen_scheduler:
            gen_checkpoint['gen_scheduler_state_dict'] = self.gen_scheduler.state_dict()
        
        # Save JEPA model checkpoint
        jepa_checkpoint = {
            'epoch': epoch,
            'jepa_state_encoder_state_dict': self.jepa_state_encoder.state_dict(),
            'jepa_target_encoder_state_dict': self.jepa_target_encoder.state_dict(),
            'jepa_action_encoder_state_dict': self.jepa_action_encoder.state_dict(),
            'jepa_transition_model_state_dict': self.jepa_transition_model.state_dict(),
            'jepa_optimizer_state_dict': self.jepa_optimizer.state_dict(),
            'jepa_metrics': metrics['jepa'],
            'config': self.config
        }
        
        if self.jepa_scheduler:
            jepa_checkpoint['jepa_scheduler_state_dict'] = self.jepa_scheduler.state_dict()
        
        # Save latest checkpoints
        gen_checkpoint_path = self.output_dir / f'checkpoint_generative_epoch_{epoch+1}.pt'
        jepa_checkpoint_path = self.output_dir / f'checkpoint_jepa_epoch_{epoch+1}.pt'
        torch.save(gen_checkpoint, gen_checkpoint_path)
        torch.save(jepa_checkpoint, jepa_checkpoint_path)
        
        # Save best models
        if is_best_gen:
            best_gen_path = self.output_dir / 'best_generative_model.pt'
            torch.save(gen_checkpoint, best_gen_path)
            self.logger.info(f"New best generative model saved with validation loss: {metrics['generative']['total_loss']:.4f}")
        
        if is_best_jepa:
            best_jepa_path = self.output_dir / 'best_jepa_model.pt'
            torch.save(jepa_checkpoint, best_jepa_path)
            self.logger.info(f"New best JEPA model saved with validation loss: {metrics['jepa']['prediction_loss']:.4f}")
    
    def train(self):
        """Main training loop."""
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        patience = training_config.get('patience', 10)
        
        # Initialize wandb if specified
        if training_config.get('use_wandb', False):
            wandb.init(
                project=training_config.get('wandb_project', 'combined-world-model'),
                config=self.config,
                name=training_config.get('wandb_run_name', None)
            )
            use_wandb = True
        else:
            use_wandb = False
        
        # Track best validation losses
        best_gen_val_loss = float('inf')
        best_jepa_val_loss = float('inf')
        epochs_no_improve_gen = 0
        epochs_no_improve_jepa = 0
        
        self.logger.info("Starting combined training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Gen Train Total: {train_metrics['generative']['total_loss']:.4f}, "
                f"Gen Val Total: {val_metrics['generative']['total_loss']:.4f}, "
                f"JEPA Train Pred: {train_metrics['jepa']['prediction_loss']:.4f}, "
                f"JEPA Val Pred: {val_metrics['jepa']['prediction_loss']:.4f}, "
                f"JEPA Val Cos: {val_metrics['jepa']['cosine_sim']:.4f} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Wandb logging
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/gen_total_loss': train_metrics['generative']['total_loss'],
                    'train/gen_transition_loss': train_metrics['generative']['transition_loss'],
                    'train/gen_action_loss': train_metrics['generative']['action_loss'],
                    'train/jepa_prediction_loss': train_metrics['jepa']['prediction_loss'],
                    'train/jepa_cosine_sim': train_metrics['jepa']['cosine_sim'],
                    'val/gen_total_loss': val_metrics['generative']['total_loss'],
                    'val/gen_transition_loss': val_metrics['generative']['transition_loss'],
                    'val/gen_action_loss': val_metrics['generative']['action_loss'],
                    'val/jepa_prediction_loss': val_metrics['jepa']['prediction_loss'],
                    'val/jepa_cosine_sim': val_metrics['jepa']['cosine_sim'],
                    'time/epoch_time': epoch_time
                })
            
            # Save metrics
            for key in ['transition_loss', 'action_loss', 'total_loss']:
                self.train_metrics['generative'][key].append(train_metrics['generative'][key])
                self.val_metrics['generative'][key].append(val_metrics['generative'][key])
            
            for key in ['prediction_loss', 'cosine_sim']:
                self.train_metrics['jepa'][key].append(train_metrics['jepa'][key])
                self.val_metrics['jepa'][key].append(val_metrics['jepa'][key])
            
            # Early stopping and checkpointing
            is_best_gen = val_metrics['generative']['total_loss'] < best_gen_val_loss
            is_best_jepa = val_metrics['jepa']['prediction_loss'] < best_jepa_val_loss
            
            if is_best_gen:
                best_gen_val_loss = val_metrics['generative']['total_loss']
                epochs_no_improve_gen = 0
            else:
                epochs_no_improve_gen += 1
                
            if is_best_jepa:
                best_jepa_val_loss = val_metrics['jepa']['prediction_loss']
                epochs_no_improve_jepa = 0
            else:
                epochs_no_improve_jepa += 1
            
            # Save checkpoint
            if (epoch + 1) % training_config.get('save_interval', 5) == 0 or is_best_gen or is_best_jepa:
                self.save_checkpoint(epoch, val_metrics, is_best_gen, is_best_jepa)
            
            # Early stopping (stop when both models have stopped improving)
            if epochs_no_improve_gen >= patience and epochs_no_improve_jepa >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Combined training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation losses - Gen: {best_gen_val_loss:.4f}, JEPA: {best_jepa_val_loss:.4f}")
        
        if use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Combined Generative and JEPA World Models')
    parser.add_argument('--config', type=str, default='configs/train_combined_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer and start training
    trainer = CombinedWorldModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 