#!/usr/bin/env python3
"""
Reward Predictor Evaluation Script

This script loads a trained reward predictor and evaluates it on data or performs inference.

Usage:
    python -m src.scripts.evaluate_reward_predictor --checkpoint path/to/checkpoint.pt --config configs/train_reward_config.yaml
"""

import torch
import torch.nn.functional as F
import yaml
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import the models
from src.models.ARC_specific.ARC_state_encoder import ARC_StateEncoder
from src.models.ARC_specific.ARC_reward_predictor import RewardPredictor
from src.data.replay_buffer_dataset import ReplayBufferDataset


class RewardPredictorEvaluator:
    """
    Evaluator for trained reward predictor models.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str):
        self.config = self.load_config(config_path)
        
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
        
        # Load models and checkpoint
        self.load_models(checkpoint_path)
        
        # Setup dataset
        self.setup_dataset()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def load_models(self, checkpoint_path: str):
        """Load trained models from checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_config = checkpoint.get('config', self.config)['model']
        
        # Initialize state encoder
        self.state_encoder = ARC_StateEncoder(
            image_size=model_config['image_size'],
            input_channels=model_config['input_channels'],
            latent_dim=model_config['latent_dim_state'],
            encoder_params=model_config.get('encoder_params', {})
        ).to(self.device)
        
        # Initialize reward predictor
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
        
        # Load state dicts
        if 'state_encoder_state_dict' in checkpoint:
            self.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
        elif model_config.get('pretrained_state_encoder_path'):
            # Load from separate file if specified
            encoder_checkpoint = torch.load(model_config['pretrained_state_encoder_path'], 
                                          map_location=self.device)
            if 'state_encoder' in encoder_checkpoint:
                self.state_encoder.load_state_dict(encoder_checkpoint['state_encoder'])
            else:
                self.state_encoder.load_state_dict(encoder_checkpoint)
        
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor_state_dict'])
        
        # Set to evaluation mode
        self.state_encoder.eval()
        self.reward_predictor.eval()
        
        print("Models loaded successfully")
        print(f"Checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'metrics' in checkpoint:
            print(f"Validation loss: {checkpoint['metrics'].get('reward_loss', 'unknown'):.4f}")
    
    def setup_dataset(self):
        """Setup evaluation dataset."""
        data_config = self.config['data']
        
        # Create dataset
        self.dataset = ReplayBufferDataset(
            buffer_path=data_config['buffer_path'],
            state_shape=tuple(data_config['state_shape']),
            mode=data_config.get('mode', 'color_only'),
            num_samples=data_config.get('num_samples', None)
        )
        
        # Create data loader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0  # Set to 0 for evaluation
        )
        
        print(f"Evaluation dataset loaded with {len(self.dataset)} samples")
    
    def encode_states(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode states using the state encoder."""
        with torch.no_grad():
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
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the full dataset."""
        print("Evaluating model...")
        
        all_predictions = []
        all_targets = []
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                # Get state embeddings
                z_t, z_tp1, z_goal = self.encode_states(batch)
                
                # Get true rewards
                r_true = batch['reward'].to(self.device)
                
                # Predict rewards
                r_pred, loss = self.reward_predictor(z_t, z_tp1, z_goal, r_true)
                
                # Collect predictions and targets
                all_predictions.extend(r_pred.cpu().numpy().flatten())
                all_targets.extend(r_true.cpu().numpy().flatten())
                
                # Update metrics
                total_loss += loss.item() * len(r_true)
                num_samples += len(r_true)
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Compute metrics
        mse = total_loss / num_samples
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # Correlation
        correlation = np.corrcoef(predictions, targets)[0, 1]
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'r2_score': r2,
            'num_samples': num_samples
        }
        
        return metrics, predictions, targets
    
    def predict_single(self, state, next_state, target_state, 
                      state_metadata, next_metadata, target_metadata) -> float:
        """
        Predict reward for a single transition.
        
        Args:
            state: Current state tensor (H, W)
            next_state: Next state tensor (H, W)
            target_state: Goal state tensor (H, W)
            state_metadata: Dict with current state metadata
            next_metadata: Dict with next state metadata  
            target_metadata: Dict with target state metadata
            
        Returns:
            Predicted reward (scalar)
        """
        with torch.no_grad():
            # Prepare batch of size 1
            batch = {
                'state': state.unsqueeze(0),
                'next_state': next_state.unsqueeze(0),
                'target_state': target_state.unsqueeze(0),
                'shape_h': torch.tensor([state_metadata['shape_h']]),
                'shape_w': torch.tensor([state_metadata['shape_w']]),
                'most_present_color': torch.tensor([state_metadata['most_present_color']]),
                'least_present_color': torch.tensor([state_metadata['least_present_color']]),
                'num_colors_grid': torch.tensor([state_metadata['num_colors_grid']]),
                'shape_h_next': torch.tensor([next_metadata['shape_h']]),
                'shape_w_next': torch.tensor([next_metadata['shape_w']]),
                'most_present_color_next': torch.tensor([next_metadata['most_present_color']]),
                'least_present_color_next': torch.tensor([next_metadata['least_present_color']]),
                'num_colors_grid_next': torch.tensor([next_metadata['num_colors_grid']]),
                'shape_h_target': torch.tensor([target_metadata['shape_h']]),
                'shape_w_target': torch.tensor([target_metadata['shape_w']]),
                'most_present_color_target': torch.tensor([target_metadata['most_present_color']]),
                'least_present_color_target': torch.tensor([target_metadata['least_present_color']]),
                'num_colors_grid_target': torch.tensor([target_metadata['num_colors_grid']]),
            }
            
            # Get embeddings
            z_t, z_tp1, z_goal = self.encode_states(batch)
            
            # Predict reward
            r_pred, _ = self.reward_predictor(z_t, z_tp1, z_goal)
            
            return r_pred.item()
    
    def plot_results(self, predictions: np.ndarray, targets: np.ndarray, save_path: str = None):
        """Plot evaluation results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot of predictions vs targets
        axes[0, 0].scatter(targets, predictions, alpha=0.6, s=20)
        axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Rewards')
        axes[0, 0].set_ylabel('Predicted Rewards')
        axes[0, 0].set_title('Predictions vs True Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of prediction errors
        errors = predictions - targets
        axes[0, 1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Prediction Errors')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Distribution of true rewards
        axes[1, 0].hist(targets, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_xlabel('True Rewards')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of True Rewards')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of predicted rewards
        axes[1, 1].hist(predictions, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Predicted Rewards')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Predicted Rewards')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results plot saved to {save_path}")
        
        plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Reward Predictor')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/train_reward_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='Path to save plots (optional)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = RewardPredictorEvaluator(args.checkpoint, args.config)
    
    # Run evaluation
    metrics, predictions, targets = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {metrics['num_samples']:,}")
    print(f"Mean Squared Error: {metrics['mse']:.6f}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Squared Error: {metrics['rmse']:.6f}")
    print(f"Correlation: {metrics['correlation']:.6f}")
    print(f"R² Score: {metrics['r2_score']:.6f}")
    print("="*50)
    
    # Generate plots if requested
    if args.plot:
        evaluator.plot_results(predictions, targets, args.save_plots)


if __name__ == "__main__":
    main() 