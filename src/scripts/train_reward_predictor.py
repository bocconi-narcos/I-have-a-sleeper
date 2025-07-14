#!/usr/bin/env python3
"""
Reward Predictor Training Script

This script implements a training pipeline for the RewardPredictor that:
- Loads data from the replay buffer
- Uses a pre-trained state encoder to get embeddings
- Trains the RewardPredictor to predict rewards from current, next, and goal state embeddings

Usage:
    python -m src.scripts.train_reward_predictor --config configs/train_reward_config.yaml
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
from tqdm import tqdm
import wandb

# Import the models
from src.models.ARC_specific.ARC_state_encoder import ARC_StateEncoder
from src.models.ARC_specific.ARC_reward_predictor import RewardPredictor
from src.data.replay_buffer_dataset import ReplayBufferDataset


class RewardPredictorTrainer:
    """
    Trainer class for the reward predictor model.
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
        
        # Initialize optimizer
        self.setup_optimizer()
        
        # Initialize datasets
        self.setup_datasets()
        
        # Training metrics
        self.train_metrics = {'reward_loss': [], 'reward_mae': []}
        self.val_metrics = {'reward_loss': [], 'reward_mae': []}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / 'reward_training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Reward predictor training started. Output directory: {self.output_dir}")
        
    def setup_models(self):
        """Initialize all model components."""
        model_config = self.config['model']
        
        # State encoder - load pre-trained if specified
        self.state_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        # Load pre-trained state encoder if specified
        if 'pretrained_state_encoder_path' in model_config:
            encoder_path = model_config['pretrained_state_encoder_path']
            if os.path.exists(encoder_path):
                self.logger.info(f"Loading pre-trained state encoder from {encoder_path}")
                checkpoint = torch.load(encoder_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'state_encoder' in checkpoint:
                    state_dict = checkpoint['state_encoder']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                self.state_encoder.load_state_dict(state_dict)
                self.logger.info("Pre-trained state encoder loaded successfully")
            else:
                self.logger.warning(f"Pre-trained encoder path {encoder_path} not found. Using random initialization.")
        
        # Freeze state encoder if specified
        if model_config.get('freeze_state_encoder', True):
            for param in self.state_encoder.parameters():
                param.requires_grad = False
            self.state_encoder.eval()
            self.logger.info("State encoder frozen - weights will not be updated during training")
        
        # Reward predictor
        reward_config = model_config.get('reward_predictor', {})
        self.reward_predictor = RewardPredictor(
            d_model=model_config['latent_dim_state'],
            n_heads=reward_config.get('n_heads', 8),
            num_layers=reward_config.get('num_layers', 4),
            dim_ff=reward_config.get('dim_ff', 512),
            dropout=reward_config.get('dropout', 0.1),
            use_positional_encoding=reward_config.get('use_positional_encoding', True),
            pooling_method=reward_config.get('pooling_method', 'mean')
        ).to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.reward_predictor.parameters())
        trainable_params = sum(p.numel() for p in self.reward_predictor.parameters() if p.requires_grad)
        self.logger.info(f"Reward predictor parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        if not model_config.get('freeze_state_encoder', True):
            encoder_params = sum(p.numel() for p in self.state_encoder.parameters() if p.requires_grad)
            self.logger.info(f"State encoder trainable parameters: {encoder_params:,}")
    
    def setup_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        training_config = self.config['training']
        
        # Collect trainable parameters
        params = list(self.reward_predictor.parameters())
        if not self.config['model'].get('freeze_state_encoder', True):
            params.extend(list(self.state_encoder.parameters()))
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        if training_config.get('use_scheduler', True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=training_config.get('lr_factor', 0.5),
                patience=training_config.get('lr_patience', 5),
                verbose=True
            )
        else:
            self.scheduler = None
    
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
        val_ratio = training_config.get('val_ratio', 0.2)
        val_size = int(val_ratio * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
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
    
    def encode_states(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode states using the state encoder.
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            z_t: Current state embeddings (B, L, D)
            z_tp1: Next state embeddings (B, L, D) 
            z_goal: Goal state embeddings (B, L, D)
        """
        with torch.set_grad_enabled(not self.config['model'].get('freeze_state_encoder', True)):
            # Encode current state
            z_t = self.state_encoder(
                batch['state'].to(self.device),
                batch['shape_h'].to(self.device),
                batch['shape_w'].to(self.device),
                batch['most_present_color'].to(self.device),
                batch['least_present_color'].to(self.device),
                batch['num_colors_grid'].to(self.device)
            )
            
            # Encode next state
            z_tp1 = self.state_encoder(
                batch['next_state'].to(self.device),
                batch['shape_h_next'].to(self.device),
                batch['shape_w_next'].to(self.device),
                batch['most_present_color_next'].to(self.device),
                batch['least_present_color_next'].to(self.device),
                batch['num_colors_grid_next'].to(self.device)
            )
            
            # Encode goal/target state
            z_goal = self.state_encoder(
                batch['target_state'].to(self.device),
                batch['shape_h_target'].to(self.device),
                batch['shape_w_target'].to(self.device),
                batch['most_present_color_target'].to(self.device),
                batch['least_present_color_target'].to(self.device),
                batch['num_colors_grid_target'].to(self.device)
            )
        
        # Convert from (B, D) to (B, 1, D) to create sequence length dimension
        z_t = z_t.unsqueeze(1)
        z_tp1 = z_tp1.unsqueeze(1) 
        z_goal = z_goal.unsqueeze(1)
        
        return z_t, z_tp1, z_goal
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the reward prediction loss.
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            loss: Reward prediction loss
            metrics: Dictionary of computed metrics
        """
        # Get state embeddings
        z_t, z_tp1, z_goal = self.encode_states(batch)
        
        # Get true rewards
        r_true = batch['reward'].to(self.device)
        
        # Predict rewards
        r_pred, loss = self.reward_predictor(z_t, z_tp1, z_goal, r_true)
        
        # Compute additional metrics
        with torch.no_grad():
            mae = F.l1_loss(r_pred.squeeze(-1), r_true)
            
        metrics = {
            'reward_loss': loss.item(),
            'reward_mae': mae.item(),
        }
        
        return loss, metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.reward_predictor.train()
        if not self.config['model'].get('freeze_state_encoder', True):
            self.state_encoder.train()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, metrics = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.reward_predictor.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += metrics['reward_loss']
            total_mae += metrics['reward_mae']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{metrics['reward_loss']:.4f}",
                'MAE': f"{metrics['reward_mae']:.4f}"
            })
            
            # Log to wandb
            if hasattr(self, 'use_wandb') and self.use_wandb:
                wandb.log({
                    'train/batch_loss': metrics['reward_loss'],
                    'train/batch_mae': metrics['reward_mae'],
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Compute epoch averages
        avg_metrics = {
            'reward_loss': total_loss / num_batches,
            'reward_mae': total_mae / num_batches
        }
        
        return avg_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.reward_predictor.eval()
        self.state_encoder.eval()
        
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Compute loss
                loss, metrics = self.compute_loss(batch)
                
                # Update metrics
                total_loss += metrics['reward_loss']
                total_mae += metrics['reward_mae']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{metrics['reward_loss']:.4f}",
                    'MAE': f"{metrics['reward_mae']:.4f}"
                })
        
        # Compute epoch averages
        avg_metrics = {
            'reward_loss': total_loss / num_batches,
            'reward_mae': total_mae / num_batches
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'reward_predictor_state_dict': self.reward_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if not self.config['model'].get('freeze_state_encoder', True):
            checkpoint['state_encoder_state_dict'] = self.state_encoder.state_dict()
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_reward_predictor.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {metrics['reward_loss']:.4f}")
    
    def train(self):
        """Main training loop."""
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        patience = training_config.get('patience', 10)
        
        # Initialize wandb if specified
        if training_config.get('use_wandb', False):
            wandb.init(
                project=training_config.get('wandb_project', 'reward-predictor'),
                config=self.config,
                name=training_config.get('wandb_run_name', None)
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_metrics['reward_loss'])
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['reward_loss']:.4f}, "
                f"Train MAE: {train_metrics['reward_mae']:.4f}, "
                f"Val Loss: {val_metrics['reward_loss']:.4f}, "
                f"Val MAE: {val_metrics['reward_mae']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['reward_loss'],
                    'train/epoch_mae': train_metrics['reward_mae'],
                    'val/epoch_loss': val_metrics['reward_loss'],
                    'val/epoch_mae': val_metrics['reward_mae'],
                    'time/epoch_time': epoch_time
                })
            
            # Save metrics
            self.train_metrics['reward_loss'].append(train_metrics['reward_loss'])
            self.train_metrics['reward_mae'].append(train_metrics['reward_mae'])
            self.val_metrics['reward_loss'].append(val_metrics['reward_loss'])
            self.val_metrics['reward_mae'].append(val_metrics['reward_mae'])
            
            # Early stopping and checkpointing
            is_best = val_metrics['reward_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['reward_loss']
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            # Save checkpoint
            if (epoch + 1) % training_config.get('save_interval', 5) == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if epochs_no_improve >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train Reward Predictor')
    parser.add_argument('--config', type=str, default='configs/train_reward_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer and start training
    trainer = RewardPredictorTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main() 