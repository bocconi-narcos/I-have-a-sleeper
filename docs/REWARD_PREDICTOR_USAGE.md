# Reward Predictor Usage Guide

## Overview

The RewardPredictor system trains **two separate models** to predict scalar rewards from sequences of state embeddings. One model uses a **generative encoder** and another uses a **JEPA encoder**, allowing comparison of different representation learning approaches. Both use Transformer-based architectures with type embeddings and pooling (no CLS token).

## Architecture

- **Input**: Three state embedding sequences (z_t, z_tp1, z_goal)
- **Type Embeddings**: Learnable embeddings to distinguish sequence types
- **Transformer Encoder**: Multi-layer attention mechanism
- **Pooling**: Mean or max pooling for sequence aggregation
- **MLP Head**: Two-layer network for scalar reward prediction

## Files Structure

```
src/models/ARC_specific/ARC_reward_predictor.py  # Main RewardPredictor class
src/scripts/train_reward_predictor.py            # Training script
src/scripts/evaluate_reward_predictor.py         # Evaluation script
configs/train_reward_config.yaml                # Configuration file
```

## Quick Start

### 1. Training

```bash
# Train with default configuration
python -m src.scripts.train_reward_predictor

# Train with custom config
python -m src.scripts.train_reward_predictor --config configs/train_reward_config.yaml
```

### 2. Evaluation

```bash
# Evaluate trained model
python -m src.scripts.evaluate_reward_predictor \
    --checkpoint outputs/reward_predictor_training/best_reward_predictor.pt \
    --config configs/train_reward_config.yaml \
    --plot

# Save evaluation plots
python -m src.scripts.evaluate_reward_predictor \
    --checkpoint outputs/reward_predictor_training/best_reward_predictor.pt \
    --plot --save_plots evaluation_results.png
```

### 3. Inference

```python
import torch
from src.models.ARC_specific.ARC_reward_predictor import RewardPredictor
from src.models.ARC_specific.ARC_state_encoder import ARC_StateEncoder

# Load models
state_encoder = ARC_StateEncoder(...)
reward_predictor = RewardPredictor(d_model=128, num_layers=4)

# Get state embeddings (B, D) -> (B, 1, D)
z_t = state_encoder(...).unsqueeze(1)
z_tp1 = state_encoder(...).unsqueeze(1)  
z_goal = state_encoder(...).unsqueeze(1)

# Predict reward
r_pred, _ = reward_predictor(z_t, z_tp1, z_goal)
print(f"Predicted reward: {r_pred.item()}")
```

## Configuration

Key parameters in `configs/train_reward_config.yaml`:

```yaml
model:
  latent_dim_state: 128          # State embedding dimension
  freeze_state_encoder: true     # Both encoders will be frozen
  
  # Dual encoder paths (both required)
  generative_encoder_path: "outputs/world_model_training/best_model.pt"
  jepa_encoder_path: "outputs/world_model_training/best_jepa_model.pt"
  
  reward_predictor:
    n_heads: 8                   # Number of attention heads
    num_layers: 4                # Number of transformer layers
    dim_ff: 512                  # Feed-forward dimension
    pooling_method: "mean"       # "mean" or "max" pooling

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  patience: 10                   # Early stopping patience (per model)

data:
  buffer_path: "data/rb_challenge_joint_1000010_default.pt"
  val_ratio: 0.2
```

## Key Features

### ✅ **Dual Model Training**
- Simultaneous training of generative and JEPA-based reward predictors
- Separate optimizers, schedulers, and checkpointing for each model
- Independent early stopping for both models

### ✅ **Type Embeddings**
- Automatic type identification for current/next/goal sequences
- Learnable embeddings added to state representations

### ✅ **Flexible Architecture**  
- Configurable transformer depth and attention heads
- Choice of mean or max pooling
- Optional positional encoding

### ✅ **Training Pipeline**
- Frozen pre-trained encoder integration (both generative and JEPA)
- Comprehensive logging and checkpointing for both models
- Independent early stopping and learning rate scheduling
- Optional Weights & Biases integration with separate metrics

### ✅ **Model Comparison**
- Direct comparison between generative and JEPA representations
- Separate best model saving: `best_reward_predictor_generative.pt` and `best_reward_pred_JEPA.pt`
- Parallel validation metrics for performance analysis

## Expected Performance

Both models should learn to predict rewards by:
1. Understanding state transitions (current → next)
2. Measuring progress toward goals (next vs. goal)
3. Capturing temporal relationships in sequences

The comparison will reveal:
- **Generative representations**: May capture more detailed state information
- **JEPA representations**: May focus more on relevant dynamics and relationships
- **Performance differences**: Which approach works better for reward prediction in ARC tasks

## Integration

The RewardPredictor integrates seamlessly with:
- **ReplayBufferDataset**: Automatically loads state sequences and rewards
- **ARC_StateEncoder**: Uses pre-trained encoders for consistent representations
- **Existing training infrastructure**: Follows project patterns for configs and logging

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in config
2. **No improvement**: Check `learning_rate` and `freeze_state_encoder` settings
3. **NaN losses**: Ensure proper data normalization and gradient clipping

### Performance Tips

1. **Use pre-trained state encoder**: Set `pretrained_state_encoder_path` in config
2. **Freeze encoder initially**: Set `freeze_state_encoder: true` for stability
3. **Monitor correlation**: High correlation indicates good reward prediction
4. **Experiment with pooling**: Try both "mean" and "max" pooling methods

## Next Steps

After training a reward predictor, you can:
1. Use it in reinforcement learning for reward shaping
2. Integrate into planning algorithms for goal-conditioned tasks
3. Analyze learned representations for interpretability
4. Fine-tune on specific task domains 