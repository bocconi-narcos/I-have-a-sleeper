import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights


class ARC_ActionDecoder(nn.Module):
    """
    ActionDecoder (f̂): Continuous embedding to discrete action logits decoder.
    
    - Input: embedding E_t ∈ ℝᵈ
    - Output: logits over |A| actions
    
    Args:
        embedding_dim (int): Dimension of the input embedding d
        num_actions (int): Number of discrete actions |A|
        hidden_dims (list): List of hidden dimensions for MLP layers
        activation (str): Activation function ('relu', 'gelu')
        dropout (float): Dropout rate
        use_layer_norm (bool): Whether to use LayerNorm before final projection
    """
    
    def __init__(self,
                 embedding_dim: int,
                 num_actions: int,
                 hidden_dims: list = [512],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 use_layer_norm: bool = True):
        super().__init__()
        
        assert activation in ['relu', 'gelu'], \
            f"activation must be 'relu' or 'gelu', got {activation}"
        
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        self.use_layer_norm = use_layer_norm
        
        # Select activation function
        if activation == 'relu':
            act_fn = nn.ReLU()
        else:  # gelu
            act_fn = nn.GELU()
        
        # Build MLP layers
        layers = []
        current_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        # Optional layer normalization before final projection
        if use_layer_norm:
            layers.append(nn.LayerNorm(current_dim))
        
        # Final projection to action logits
        layers.append(nn.Linear(current_dim, num_actions))
        
        self.mlp_head = nn.Sequential(*layers)
        
        # Apply weight initialization
        self.apply(initialize_weights)
        
        # Print model statistics
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[ActionDecoder] Number of parameters: {num_params}")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ActionDecoder.
        
        Args:
            embeddings (torch.Tensor): Action embeddings of shape (B, embedding_dim) or 
                                     (B, seq_len, embedding_dim)
        
        Returns:
            torch.Tensor: Action logits of shape (B, num_actions) or (B, seq_len, num_actions)
        """
        # Validate input dimension
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, "
                           f"got {embeddings.shape[-1]}")
        
        # Pass through MLP head
        return self.mlp_head(embeddings)
