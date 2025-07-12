# JEPA World Model Training Guide

## Overview

The JEPA (Joint Embedding Predictive Architecture) world model trainer implements a simplified, efficient approach to learning world models by predicting future state embeddings rather than reconstructing pixel-level details.

## Architecture

```
Current State (s) → State Encoder φ(s) → z_s ─┐
                                              │
Action (a) → Action Encoder g(a) → z_a ──────┼─→ Transition Model T(z_s, z_a) → ẑ_s'
                                              │
Next State (s') → Target Encoder φ_target(s') → z_s' (EMA of φ)
                                              │
                                              └─→ Loss: MSE(ẑ_s', z_s')
```

## Key Features

- **Embedding-space learning**: Avoids pixel-level reconstruction complexity
- **EMA target encoder**: Prevents representation collapse  
- **Transformer-based transition model**: Learns complex dynamics
- **Efficient training**: Faster than generative approaches

## Usage

### 1. Basic Training

```bash
python -m src.scripts.train_JEPA_wm
```

### 2. Custom Configuration

```bash
python -m src.scripts.train_JEPA_wm --config custom_config.yaml
```

### 3. Configuration Structure

The configuration file `configs/train_JEPA_config.yaml` contains:

- **Model**: Architecture parameters (dimensions, depths, etc.)
- **Training**: Learning rates, EMA decay, epochs
- **Data**: Dataset paths, preprocessing options
- **Experiment**: Logging and checkpointing settings

### 4. Key Parameters

#### Critical Parameters:
- `ema_decay`: Controls stability of target encoder (default: 0.995)
- `latent_dim_state`: State embedding dimension (default: 128)
- `latent_dim_action`: Action embedding dimension (default: 64)
- `transition_depth`: Transformer layers in transition model (default: 4)

#### Training Parameters:
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.0001)
- `num_epochs`: Training epochs (default: 100)
- `patience`: Early stopping patience (default: 15)

## Output

The trainer saves:
- **Best model**: `outputs/jepa_training/best_jepa_model.pt`
- **Logs**: Training metrics via wandb
- **Checkpoints**: Periodic saves (optional)

## Monitoring

The trainer tracks:
- Training/validation loss
- Cosine similarity between predicted and target embeddings
- EMA decay effectiveness
- Training speed and convergence

## Comparison with Generative Model

| Aspect | JEPA | Generative |
|--------|------|------------|
| **Objective** | Predict embeddings | Reconstruct pixels |
| **Efficiency** | High | Lower |
| **Speed** | Fast | Slower |
| **Stability** | EMA prevents collapse | Needs careful tuning |
| **Interpretability** | Abstract representations | Pixel-level detail |

## Troubleshooting

### Common Issues:

1. **Representation collapse**: Increase `ema_decay` (closer to 1.0)
2. **Slow convergence**: Increase `learning_rate` or `batch_size`
3. **Poor performance**: Increase `latent_dim_state` or `transition_depth`
4. **Memory issues**: Reduce `batch_size` or model dimensions

### Validation:

Monitor cosine similarity between predicted and target embeddings:
- **Good**: > 0.7 after few epochs
- **Concerning**: < 0.5 after many epochs (potential collapse)

## Advanced Usage

### Custom Architectures:

Modify `configs/train_JEPA_config.yaml`:
- Change `encoder_type` to 'cnn' for CNN-based state encoder
- Adjust `transition_depth` for model capacity
- Tune `ema_decay` for stability

### Hyperparameter Tuning:

Key parameters to tune:
1. `ema_decay`: [0.99, 0.995, 0.999]
2. `learning_rate`: [1e-5, 1e-4, 1e-3]
3. `latent_dim_state`: [64, 128, 256]
4. `transition_depth`: [2, 4, 6]

## Theory

JEPA learns by:
1. Encoding current state and action
2. Predicting next state embedding
3. Comparing with target encoder's next state embedding
4. Using EMA to slowly update target encoder

This approach learns rich representations without pixel-level reconstruction, making it efficient and stable for world model learning. 