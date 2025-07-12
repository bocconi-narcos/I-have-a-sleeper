#!/usr/bin/env python3
"""
Supervised Training Script for World Model

This script implements a complete training pipeline for a world model consisting of:
- State encoder phi(s) → latent state x
- Action encoder g(a) → latent action e  
- Transition model T(x, e) → prediction for next-state distribution
- Action decoder f(e) → reconstruction of original action a

Usage:
    python -m src.scripts.train_generative_wm --config configs/train_config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import yaml
import argparse
import os
import time
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import numpy as np

# Import the existing models
from src.models.ARC_specific.ARC_state_encoder import ARC_StateEncoder
from src.models.ARC_specific.ARC_state_decoder import ARC_StateDecoder
from src.models.ARC_specific.ARC_action_encoder import ARC_ActionEncoder
from src.models.ARC_specific.ARC_transition_model import ARC_TransitionModel
from src.models.ARC_specific.ARC_action_decoder import ARC_ActionDecoder
from src.data.replay_buffer_dataset import ReplayBufferDataset


class WorldModelTrainer:
    """
    Trainer class for supervised world model training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.setup_models()
        
        # Initialize optimizer
        self.setup_optimizer()
        
        # Initialize datasets
        self.setup_datasets()
        
        # Training metrics
        self.train_metrics = {'transition_loss': [], 'action_loss': [], 'total_loss': [], 'state_reconstruction_loss': []}
        self.val_metrics = {'transition_loss': [], 'action_loss': [], 'total_loss': [], 'state_reconstruction_loss': []}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training started. Output directory: {self.output_dir}")
        
    def setup_models(self):
        """Initialize all model components."""
        model_config = self.config['model']
        
        # State encoder (phi)
        self.state_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        # Action encoder (g)
        self.action_encoder = ARC_ActionEncoder(
            num_actions=model_config['num_actions'],
            embedding_dim=model_config['latent_dim_action'],
            encoder_type=model_config.get('action_encoder_type', 'embedding')
        ).to(self.device)
        
        # Transition model (T)
        self.transition_model = ARC_TransitionModel(
            state_dim=model_config['latent_dim_state'],
            action_dim=model_config['latent_dim_action'],
            hidden_dim=model_config.get('transition_hidden_dim', 256),
            dropout=model_config.get('dropout', 0.0)
        ).to(self.device)
        
        # Action decoder (f)
        self.action_decoder = ARC_ActionDecoder(
            embedding_dim=model_config['latent_dim_action'],
            num_actions=model_config['num_actions'],
            hidden_dims=model_config.get('action_decoder_hidden_dims', [512]),
            dropout=model_config.get('dropout', 0.0)
        ).to(self.device)
        
        # State decoder (reconstructs states from latent embeddings)
        self.state_decoder = ARC_StateDecoder(
            image_size=model_config['image_size'],
            latent_dim=model_config['latent_dim_state'],
            decoder_params=model_config.get('decoder_params', {})
        ).to(self.device)
        
        # Log model parameters
        total_params = sum(p.numel() for p in self.state_encoder.parameters()) + \
                      sum(p.numel() for p in self.action_encoder.parameters()) + \
                      sum(p.numel() for p in self.transition_model.parameters()) + \
                      sum(p.numel() for p in self.action_decoder.parameters()) + \
                      sum(p.numel() for p in self.state_decoder.parameters())
        self.logger.info(f"Total model parameters: {total_params:,}")
        
    def setup_optimizer(self):
        """Initialize optimizer for all model parameters."""
        # Collect all parameters
        all_params = list(self.state_encoder.parameters()) + \
                    list(self.action_encoder.parameters()) + \
                    list(self.transition_model.parameters()) + \
                    list(self.action_decoder.parameters()) + \
                    list(self.state_decoder.parameters())
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 1e-4)
        )
        
        # Optional learning rate scheduler
        if self.config['training'].get('use_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training'].get('scheduler_step_size', 10),
                gamma=self.config['training'].get('scheduler_gamma', 0.1)
            )
        else:
            self.scheduler = None
            
    def setup_datasets(self):
        """Initialize datasets and dataloaders."""
        data_config = self.config['data']
        
        # Create full dataset
        full_dataset = ReplayBufferDataset(
            buffer_path=data_config['buffer_path'],
            state_shape=tuple(data_config['state_shape']),
            mode=data_config.get('mode', 'color_only'),
            num_samples=data_config.get('num_samples', None)
        )
        
        # Split into train and validation
        train_size = int(data_config['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'mps' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'mps' else False
        )
        
        self.logger.info(f"Dataset split: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
        
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute transition, action reconstruction, and state reconstruction losses.
        
        Args:
            batch: Dictionary containing batch data
            
        Returns:
            Tuple of (total_loss, transition_loss, action_loss, state_reconstruction_loss)
        """
        # Move batch to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
        
        # Encode current state
        x_t = self.state_encoder(
            batch['state'],
            batch['shape_h'],
            batch['shape_w'],
            batch['most_present_color'],
            batch['least_present_color'],
            batch['num_colors_grid']
        )
        
        # Encode actions
        e_t = self.action_encoder(batch['action'])
        
        # Predict next state embedding using transition model
        x_next_pred = self.transition_model.predict(x_t, e_t)
        
        # Decode predicted next state
        decoded_next_state = self.state_decoder(x_next_pred)
        
        # Compute state reconstruction loss
        state_reconstruction_loss = self._compute_state_reconstruction_loss(
            decoded_next_state, batch
        )
        
        # Optional: Also compute transition loss in embedding space for comparison
        # (can be disabled by setting weight to 0 in config)
        x_next_true = self.state_encoder(
            batch['next_state'],
            batch['shape_h_next'],
            batch['shape_w_next'],
            batch['most_present_color_next'],
            batch['least_present_color_next'],
            batch['num_colors_grid_next']
        )
        embedding_transition_loss = self.transition_model.nll_loss(x_t, e_t, x_next_true)
        
        # Compute action reconstruction loss
        action_logits = self.action_decoder(e_t)
        
        if self.config['training']['loss_type'] == 'categorical':
            action_loss = F.cross_entropy(action_logits, batch['action'])
        else:  # MSE for continuous actions
            action_loss = F.mse_loss(action_logits, batch['action'].float())
        
        # Combine losses with configurable weights
        alpha = self.config['training']['action_loss_weight']
        beta = self.config['training'].get('state_reconstruction_loss_weight', 1.0)
        gamma = self.config['training'].get('embedding_transition_loss_weight', 0.1)
        
        total_loss = (beta * state_reconstruction_loss + 
                     alpha * action_loss + 
                     gamma * embedding_transition_loss)
        
        return total_loss, embedding_transition_loss, action_loss, state_reconstruction_loss
    
    def _compute_state_reconstruction_loss(self, decoded_state: Dict[str, torch.Tensor], 
                                         batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute state reconstruction loss by comparing decoded state with true next state.
        
        Args:
            decoded_state: Dictionary of decoded state logits from state decoder
            batch: Batch data containing true next state
            
        Returns:
            torch.Tensor: State reconstruction loss
        """
        total_loss = 0.0
        
        # Grid reconstruction loss
        grid_logits = decoded_state['grid_logits']  # (B, H, W, vocab_size)
        true_grid = batch['next_state']  # (B, H, W) or (B, 1, H, W)
        
        # Handle potential channel dimension
        if true_grid.dim() == 4 and true_grid.shape[1] == 1:
            true_grid = true_grid.squeeze(1)  # (B, H, W)
        
        # Flatten for cross-entropy
        B, H, W = true_grid.shape
        grid_logits_flat = grid_logits.view(B * H * W, -1)
        true_grid_flat = true_grid.view(B * H * W)
        
        # Handle padding (-1 values) by masking
        mask = true_grid_flat != -1
        if mask.any():
            grid_logits_masked = grid_logits_flat[mask]
            true_grid_masked = true_grid_flat[mask] + 1  # Shift to match embedding shift
            grid_loss = F.cross_entropy(grid_logits_masked, true_grid_masked)
        else:
            grid_loss = torch.tensor(0.0, device=self.device)
        
        total_loss += grid_loss
        
        # Shape reconstruction losses
        shape_h_loss = F.cross_entropy(decoded_state['shape_h_logits'], batch['shape_h_next'] - 1)
        shape_w_loss = F.cross_entropy(decoded_state['shape_w_logits'], batch['shape_w_next'] - 1)
        total_loss += shape_h_loss + shape_w_loss
        
        # Color statistics reconstruction losses
        most_common_loss = F.cross_entropy(decoded_state['most_common_logits'], 
                                          batch['most_present_color_next'])
        least_common_loss = F.cross_entropy(decoded_state['least_common_logits'], 
                                           batch['least_present_color_next'])
        unique_count_loss = F.cross_entropy(decoded_state['unique_count_logits'], 
                                           batch['num_colors_grid_next'])
        
        total_loss += most_common_loss + least_common_loss + unique_count_loss
        
        return total_loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.state_encoder.train()
        self.action_encoder.train()
        self.transition_model.train()
        self.action_decoder.train()
        self.state_decoder.train()
        
        epoch_losses = {'transition_loss': [], 'action_loss': [], 'total_loss': [], 'state_reconstruction_loss': []}
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute losses
            total_loss, transition_loss, action_loss, state_reconstruction_loss = self.compute_losses(batch)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.state_encoder.parameters()) + \
                    list(self.action_encoder.parameters()) + \
                    list(self.transition_model.parameters()) + \
                    list(self.action_decoder.parameters()) + \
                    list(self.state_decoder.parameters()),
                    self.config['training']['grad_clip']
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Store losses
            epoch_losses['total_loss'].append(total_loss.item())
            epoch_losses['transition_loss'].append(transition_loss.item())
            epoch_losses['action_loss'].append(action_loss.item())
            epoch_losses['state_reconstruction_loss'].append(state_reconstruction_loss.item())
            
            # Log progress
            if batch_idx % self.config['training'].get('log_interval', 100) == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}: '
                    f'Total Loss: {total_loss.item():.4f}, '
                    f'Transition Loss: {transition_loss.item():.4f}, '
                    f'Action Loss: {action_loss.item():.4f}, '
                    f'State Reconstruction Loss: {state_reconstruction_loss.item():.4f}'
                )
        
        # Compute average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average validation losses
        """
        self.state_encoder.eval()
        self.action_encoder.eval()
        self.transition_model.eval()
        self.action_decoder.eval()
        self.state_decoder.eval()
        
        epoch_losses = {'transition_loss': [], 'action_loss': [], 'total_loss': [], 'state_reconstruction_loss': []}
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Compute losses
                total_loss, transition_loss, action_loss, state_reconstruction_loss = self.compute_losses(batch)
                
                # Store losses
                epoch_losses['total_loss'].append(total_loss.item())
                epoch_losses['transition_loss'].append(transition_loss.item())
                epoch_losses['action_loss'].append(action_loss.item())
                epoch_losses['state_reconstruction_loss'].append(state_reconstruction_loss.item())
        
        # Compute average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        self.logger.info(
            f'Validation Epoch {epoch}: '
            f'Total Loss: {avg_losses["total_loss"]:.4f}, '
            f'Transition Loss: {avg_losses["transition_loss"]:.4f}, '
            f'Action Loss: {avg_losses["action_loss"]:.4f}, '
            f'State Reconstruction Loss: {avg_losses["state_reconstruction_loss"]:.4f}'
        )
        
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'state_encoder_state_dict': self.state_encoder.state_dict(),
            'action_encoder_state_dict': self.action_encoder.state_dict(),
            'transition_model_state_dict': self.transition_model.state_dict(),
            'action_decoder_state_dict': self.action_decoder.state_dict(),
            'state_decoder_state_dict': self.state_decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f'Saved best model at epoch {epoch}')
        
        self.logger.info(f'Saved checkpoint: {checkpoint_path}')
    
    def train(self):
        """
        Main training loop.
        """
        self.logger.info("Starting training...")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            epoch_start_time = time.time()
            
            # Train epoch
            train_losses = self.train_epoch(epoch)
            
            # Validate epoch
            val_losses = self.validate_epoch(epoch)
            
            # Update metrics
            for key in train_losses:
                self.train_metrics[key].append(train_losses[key])
                self.val_metrics[key].append(val_losses[key])
            
            # Check if this is the best model
            is_best = val_losses['total_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_losses['total_loss']
            
            # Save checkpoint
            if epoch % self.config['training'].get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f'Epoch {epoch} completed in {epoch_time:.2f}s. '
                f'Train Loss: {train_losses["total_loss"]:.4f}, '
                f'Val Loss: {val_losses["total_loss"]:.4f}'
            )
        
        self.logger.info("Training completed!")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train World Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = WorldModelTrainer(config)
    
    # TODO: Add resume functionality if needed
    if args.resume:
        # Load checkpoint and resume training
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        trainer.action_encoder.load_state_dict(checkpoint['action_encoder_state_dict'])
        trainer.transition_model.load_state_dict(checkpoint['transition_model_state_dict'])
        trainer.action_decoder.load_state_dict(checkpoint['action_decoder_state_dict'])
        trainer.state_decoder.load_state_dict(checkpoint['state_decoder_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main() 