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
    Trainer class for dual reward predictor models using both generative and JEPA encoders.
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
            'generative': {'reward_loss': [], 'reward_mae': []},
            'jepa': {'reward_loss': [], 'reward_mae': []}
        }
        self.val_metrics = {
            'generative': {'reward_loss': [], 'reward_mae': []},
            'jepa': {'reward_loss': [], 'reward_mae': []}
        }
        
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
        
        # Initialize both generative and JEPA state encoders
        self.generative_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        self.jepa_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        # Load pre-trained generative encoder
        if 'generative_encoder_path' in model_config:
            encoder_path = model_config['generative_encoder_path']
            if os.path.exists(encoder_path):
                self.logger.info(f"Loading pre-trained generative encoder from {encoder_path}")
                checkpoint = torch.load(encoder_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'state_encoder' in checkpoint:
                    state_dict = checkpoint['state_encoder']
                elif 'state_encoder_state_dict' in checkpoint:
                    state_dict = checkpoint['state_encoder_state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                self.generative_encoder.load_state_dict(state_dict)
                self.logger.info("Pre-trained generative encoder loaded successfully")
            else:
                self.logger.warning(f"Generative encoder path {encoder_path} not found. Using random initialization.")
        
        # Load pre-trained JEPA encoder
        if 'jepa_encoder_path' in model_config:
            encoder_path = model_config['jepa_encoder_path']
            if os.path.exists(encoder_path):
                self.logger.info(f"Loading pre-trained JEPA encoder from {encoder_path}")
                checkpoint = torch.load(encoder_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'state_encoder' in checkpoint:
                    state_dict = checkpoint['state_encoder']
                elif 'state_encoder_state_dict' in checkpoint:
                    state_dict = checkpoint['state_encoder_state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                self.jepa_encoder.load_state_dict(state_dict)
                self.logger.info("Pre-trained JEPA encoder loaded successfully")
            else:
                self.logger.warning(f"JEPA encoder path {encoder_path} not found. Using random initialization.")
        
        # Freeze both encoders (they should always be frozen)
        for param in self.generative_encoder.parameters():
            param.requires_grad = False
        for param in self.jepa_encoder.parameters():
            param.requires_grad = False
        
        self.generative_encoder.eval()
        self.jepa_encoder.eval()
        self.logger.info("Both encoders frozen - weights will not be updated during training")
        
        # Create two separate reward predictors
        reward_config = model_config.get('reward_predictor', {})
        
        self.reward_predictor_generative = RewardPredictor(
            d_model=model_config['latent_dim_state'],
            n_heads=reward_config.get('n_heads', 8),
            num_layers=reward_config.get('num_layers', 4),
            dim_ff=reward_config.get('dim_ff', 512),
            dropout=reward_config.get('dropout', 0.1),
            use_positional_encoding=reward_config.get('use_positional_encoding', True),
            pooling_method=reward_config.get('pooling_method', 'mean')
        ).to(self.device)
        
        self.reward_predictor_jepa = RewardPredictor(
            d_model=model_config['latent_dim_state'],
            n_heads=reward_config.get('n_heads', 8),
            num_layers=reward_config.get('num_layers', 4),
            dim_ff=reward_config.get('dim_ff', 512),
            dropout=reward_config.get('dropout', 0.1),
            use_positional_encoding=reward_config.get('use_positional_encoding', True),
            pooling_method=reward_config.get('pooling_method', 'mean')
        ).to(self.device)
        
        # Print model info
        gen_params = sum(p.numel() for p in self.reward_predictor_generative.parameters() if p.requires_grad)
        jepa_params = sum(p.numel() for p in self.reward_predictor_jepa.parameters() if p.requires_grad)
        self.logger.info(f"Generative reward predictor trainable parameters: {gen_params:,}")
        self.logger.info(f"JEPA reward predictor trainable parameters: {jepa_params:,}")
    
    def setup_optimizers(self):
        """Initialize optimizers and learning rate schedulers for both models."""
        training_config = self.config['training']
        
        # Create separate optimizers for both reward predictors
        self.optimizer_generative = torch.optim.AdamW(
            self.reward_predictor_generative.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        self.optimizer_jepa = torch.optim.AdamW(
            self.reward_predictor_jepa.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )
        
        # Learning rate schedulers
        if training_config.get('use_scheduler', True):
            self.scheduler_generative = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_generative,
                mode='min',
                factor=training_config.get('lr_factor', 0.5),
                patience=training_config.get('lr_patience', 5),
                verbose=True
            )
            self.scheduler_jepa = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_jepa,
                mode='min',
                factor=training_config.get('lr_factor', 0.5),
                patience=training_config.get('lr_patience', 5),
                verbose=True
            )
        else:
            self.scheduler_generative = None
            self.scheduler_jepa = None
    
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
    
    def encode_states_generative(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode states using the generative encoder.
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            z_t: Current state embeddings (B, L, D)
            z_tp1: Next state embeddings (B, L, D) 
            z_goal: Goal state embeddings (B, L, D)
        """
        with torch.no_grad():  # Encoders are always frozen
            # Encode current state
            z_t = self.generative_encoder(
                batch['state'].to(self.device),
                batch['shape_h'].to(self.device),
                batch['shape_w'].to(self.device),
                batch['most_present_color'].to(self.device),
                batch['least_present_color'].to(self.device),
                batch['num_colors_grid'].to(self.device)
            )
            
            # Encode next state
            z_tp1 = self.generative_encoder(
                batch['next_state'].to(self.device),
                batch['shape_h_next'].to(self.device),
                batch['shape_w_next'].to(self.device),
                batch['most_present_color_next'].to(self.device),
                batch['least_present_color_next'].to(self.device),
                batch['num_colors_grid_next'].to(self.device)
            )
            
            # Encode goal/target state
            z_goal = self.generative_encoder(
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
    
    def encode_states_jepa(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode states using the JEPA encoder.
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            z_t: Current state embeddings (B, L, D)
            z_tp1: Next state embeddings (B, L, D) 
            z_goal: Goal state embeddings (B, L, D)
        """
        with torch.no_grad():  # Encoders are always frozen
            # Encode current state
            z_t = self.jepa_encoder(
                batch['state'].to(self.device),
                batch['shape_h'].to(self.device),
                batch['shape_w'].to(self.device),
                batch['most_present_color'].to(self.device),
                batch['least_present_color'].to(self.device),
                batch['num_colors_grid'].to(self.device)
            )
            
            # Encode next state
            z_tp1 = self.jepa_encoder(
                batch['next_state'].to(self.device),
                batch['shape_h_next'].to(self.device),
                batch['shape_w_next'].to(self.device),
                batch['most_present_color_next'].to(self.device),
                batch['least_present_color_next'].to(self.device),
                batch['num_colors_grid_next'].to(self.device)
            )
            
            # Encode goal/target state
            z_goal = self.jepa_encoder(
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
    
    def compute_loss_generative(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the reward prediction loss for the generative model.
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            loss: Reward prediction loss
            metrics: Dictionary of computed metrics
        """
        # Get state embeddings
        z_t, z_tp1, z_goal = self.encode_states_generative(batch)
        
        # Get true rewards
        r_true = batch['reward'].to(self.device)
        
        # Predict rewards
        r_pred, loss = self.reward_predictor_generative(z_t, z_tp1, z_goal, r_true)
        
        # Compute additional metrics
        with torch.no_grad():
            mae = F.l1_loss(r_pred.squeeze(-1), r_true)
            
        metrics = {
            'reward_loss': loss.item(),
            'reward_mae': mae.item(),
        }
        
        return loss, metrics
    
    def compute_loss_jepa(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the reward prediction loss for the JEPA model.
        
        Args:
            batch: Batch of data from the replay buffer
            
        Returns:
            loss: Reward prediction loss
            metrics: Dictionary of computed metrics
        """
        # Get state embeddings
        z_t, z_tp1, z_goal = self.encode_states_jepa(batch)
        
        # Get true rewards
        r_true = batch['reward'].to(self.device)
        
        # Predict rewards
        r_pred, loss = self.reward_predictor_jepa(z_t, z_tp1, z_goal, r_true)
        
        # Compute additional metrics
        with torch.no_grad():
            mae = F.l1_loss(r_pred.squeeze(-1), r_true)
            
        metrics = {
            'reward_loss': loss.item(),
            'reward_mae': mae.item(),
        }
        
        return loss, metrics
    
    def train_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """Train both models for one epoch."""
        self.reward_predictor_generative.train()
        self.reward_predictor_jepa.train()
        
        # Track metrics for both models
        total_loss_gen = 0
        total_mae_gen = 0
        total_loss_jepa = 0
        total_mae_jepa = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Train generative model
            self.optimizer_generative.zero_grad()
            loss_gen, metrics_gen = self.compute_loss_generative(batch)
            loss_gen.backward()
            
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.reward_predictor_generative.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer_generative.step()
            
            # Train JEPA model
            self.optimizer_jepa.zero_grad()
            loss_jepa, metrics_jepa = self.compute_loss_jepa(batch)
            loss_jepa.backward()
            
            if self.config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.reward_predictor_jepa.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer_jepa.step()
            
            # Update metrics
            total_loss_gen += metrics_gen['reward_loss']
            total_mae_gen += metrics_gen['reward_mae']
            total_loss_jepa += metrics_jepa['reward_loss']
            total_mae_jepa += metrics_jepa['reward_mae']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Gen Loss': f"{metrics_gen['reward_loss']:.4f}",
                'JEPA Loss': f"{metrics_jepa['reward_loss']:.4f}",
                'Gen MAE': f"{metrics_gen['reward_mae']:.4f}",
                'JEPA MAE': f"{metrics_jepa['reward_mae']:.4f}"
            })
            
            # Log to wandb
            if hasattr(self, 'use_wandb') and self.use_wandb:
                wandb.log({
                    'train/batch_loss_generative': metrics_gen['reward_loss'],
                    'train/batch_mae_generative': metrics_gen['reward_mae'],
                    'train/batch_loss_jepa': metrics_jepa['reward_loss'],
                    'train/batch_mae_jepa': metrics_jepa['reward_mae'],
                    'train/lr_generative': self.optimizer_generative.param_groups[0]['lr'],
                    'train/lr_jepa': self.optimizer_jepa.param_groups[0]['lr']
                })
        
        # Compute epoch averages
        avg_metrics = {
            'generative': {
                'reward_loss': total_loss_gen / num_batches,
                'reward_mae': total_mae_gen / num_batches
            },
            'jepa': {
                'reward_loss': total_loss_jepa / num_batches,
                'reward_mae': total_mae_jepa / num_batches
            }
        }
        
        return avg_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """Validate both models for one epoch."""
        self.reward_predictor_generative.eval()
        self.reward_predictor_jepa.eval()
        
        # Track metrics for both models
        total_loss_gen = 0
        total_mae_gen = 0
        total_loss_jepa = 0
        total_mae_jepa = 0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Validate generative model
                loss_gen, metrics_gen = self.compute_loss_generative(batch)
                
                # Validate JEPA model
                loss_jepa, metrics_jepa = self.compute_loss_jepa(batch)
                
                # Update metrics
                total_loss_gen += metrics_gen['reward_loss']
                total_mae_gen += metrics_gen['reward_mae']
                total_loss_jepa += metrics_jepa['reward_loss']
                total_mae_jepa += metrics_jepa['reward_mae']
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Gen Loss': f"{metrics_gen['reward_loss']:.4f}",
                    'JEPA Loss': f"{metrics_jepa['reward_loss']:.4f}",
                    'Gen MAE': f"{metrics_gen['reward_mae']:.4f}",
                    'JEPA MAE': f"{metrics_jepa['reward_mae']:.4f}"
                })
        
        # Compute epoch averages
        avg_metrics = {
            'generative': {
                'reward_loss': total_loss_gen / num_batches,
                'reward_mae': total_mae_gen / num_batches
            },
            'jepa': {
                'reward_loss': total_loss_jepa / num_batches,
                'reward_mae': total_mae_jepa / num_batches
            }
        }
        
        return avg_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, Dict[str, float]], 
                       is_best_gen: bool = False, is_best_jepa: bool = False):
        """Save model checkpoints for both models."""
        
        # Save generative model checkpoint
        checkpoint_gen = {
            'epoch': epoch,
            'reward_predictor_state_dict': self.reward_predictor_generative.state_dict(),
            'optimizer_state_dict': self.optimizer_generative.state_dict(),
            'metrics': metrics['generative'],
            'config': self.config,
            'model_type': 'generative'
        }
        
        if self.scheduler_generative:
            checkpoint_gen['scheduler_state_dict'] = self.scheduler_generative.state_dict()
        
        # Save JEPA model checkpoint
        checkpoint_jepa = {
            'epoch': epoch,
            'reward_predictor_state_dict': self.reward_predictor_jepa.state_dict(),
            'optimizer_state_dict': self.optimizer_jepa.state_dict(),
            'metrics': metrics['jepa'],
            'config': self.config,
            'model_type': 'jepa'
        }
        
        if self.scheduler_jepa:
            checkpoint_jepa['scheduler_state_dict'] = self.scheduler_jepa.state_dict()
        
        # Save latest checkpoints
        checkpoint_path_gen = self.output_dir / f'checkpoint_generative_epoch_{epoch+1}.pt'
        checkpoint_path_jepa = self.output_dir / f'checkpoint_jepa_epoch_{epoch+1}.pt'
        torch.save(checkpoint_gen, checkpoint_path_gen)
        torch.save(checkpoint_jepa, checkpoint_path_jepa)
        
        # Save best models with specific names
        if is_best_gen:
            best_path_gen = self.output_dir / 'best_reward_predictor_generative.pt'
            torch.save(checkpoint_gen, best_path_gen)
            self.logger.info(f"New best generative model saved with validation loss: {metrics['generative']['reward_loss']:.4f}")
        
        if is_best_jepa:
            best_path_jepa = self.output_dir / 'best_reward_pred_JEPA.pt'
            torch.save(checkpoint_jepa, best_path_jepa)
            self.logger.info(f"New best JEPA model saved with validation loss: {metrics['jepa']['reward_loss']:.4f}")
    
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
        
        # Track best validation losses for both models
        best_val_loss_gen = float('inf')
        best_val_loss_jepa = float('inf')
        epochs_no_improve_gen = 0
        epochs_no_improve_jepa = 0
        
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rates
            if self.scheduler_generative:
                self.scheduler_generative.step(val_metrics['generative']['reward_loss'])
            if self.scheduler_jepa:
                self.scheduler_jepa.step(val_metrics['jepa']['reward_loss'])
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Generative - Train Loss: {train_metrics['generative']['reward_loss']:.4f}, "
                f"Train MAE: {train_metrics['generative']['reward_mae']:.4f}, "
                f"Val Loss: {val_metrics['generative']['reward_loss']:.4f}, "
                f"Val MAE: {val_metrics['generative']['reward_mae']:.4f} | "
                f"JEPA - Train Loss: {train_metrics['jepa']['reward_loss']:.4f}, "
                f"Train MAE: {train_metrics['jepa']['reward_mae']:.4f}, "
                f"Val Loss: {val_metrics['jepa']['reward_loss']:.4f}, "
                f"Val MAE: {val_metrics['jepa']['reward_mae']:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Wandb logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss_generative': train_metrics['generative']['reward_loss'],
                    'train/epoch_mae_generative': train_metrics['generative']['reward_mae'],
                    'val/epoch_loss_generative': val_metrics['generative']['reward_loss'],
                    'val/epoch_mae_generative': val_metrics['generative']['reward_mae'],
                    'train/epoch_loss_jepa': train_metrics['jepa']['reward_loss'],
                    'train/epoch_mae_jepa': train_metrics['jepa']['reward_mae'],
                    'val/epoch_loss_jepa': val_metrics['jepa']['reward_loss'],
                    'val/epoch_mae_jepa': val_metrics['jepa']['reward_mae'],
                    'time/epoch_time': epoch_time
                })
            
            # Save metrics
            self.train_metrics['generative']['reward_loss'].append(train_metrics['generative']['reward_loss'])
            self.train_metrics['generative']['reward_mae'].append(train_metrics['generative']['reward_mae'])
            self.val_metrics['generative']['reward_loss'].append(val_metrics['generative']['reward_loss'])
            self.val_metrics['generative']['reward_mae'].append(val_metrics['generative']['reward_mae'])
            
            self.train_metrics['jepa']['reward_loss'].append(train_metrics['jepa']['reward_loss'])
            self.train_metrics['jepa']['reward_mae'].append(train_metrics['jepa']['reward_mae'])
            self.val_metrics['jepa']['reward_loss'].append(val_metrics['jepa']['reward_loss'])
            self.val_metrics['jepa']['reward_mae'].append(val_metrics['jepa']['reward_mae'])
            
            # Early stopping and checkpointing for both models
            is_best_gen = val_metrics['generative']['reward_loss'] < best_val_loss_gen
            is_best_jepa = val_metrics['jepa']['reward_loss'] < best_val_loss_jepa
            
            if is_best_gen:
                best_val_loss_gen = val_metrics['generative']['reward_loss']
                epochs_no_improve_gen = 0
            else:
                epochs_no_improve_gen += 1
                
            if is_best_jepa:
                best_val_loss_jepa = val_metrics['jepa']['reward_loss']
                epochs_no_improve_jepa = 0
            else:
                epochs_no_improve_jepa += 1
            
            # Save checkpoint
            if (epoch + 1) % training_config.get('save_interval', 5) == 0 or is_best_gen or is_best_jepa:
                self.save_checkpoint(epoch, val_metrics, is_best_gen, is_best_jepa)
            
            # Early stopping (stop when both models have stopped improving)
            if epochs_no_improve_gen >= patience and epochs_no_improve_jepa >= patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs (both models stopped improving)")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation loss - Generative: {best_val_loss_gen:.4f}, JEPA: {best_val_loss_jepa:.4f}")
        
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