import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights


class ARC_ActionEncoder(nn.Module):
    """
    ActionEncoder (g): Discrete action encoder that maps integer action indices to continuous embeddings.
    
    - Input: integer action index A_t ∈ {0,...,|A|-1}
    - Output: continuous embedding E_t ∈ ℝᵈ
    
    Args:
        num_actions (int): Number of discrete actions |A|
        embedding_dim (int): Dimension of the output embedding d
        encoder_type (str): Type of encoder - 'embedding' or 'onehot_mlp'
        hidden_dim (int): Hidden dimension for MLP encoder (only used if encoder_type='onehot_mlp')
        dropout (float): Dropout rate for MLP encoder
    """
    
    def __init__(self, 
                 num_actions: int,
                 embedding_dim: int,
                 encoder_type: str = 'embedding',
                 hidden_dim: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        
        assert encoder_type in ['embedding', 'onehot_mlp'], \
            f"encoder_type must be 'embedding' or 'onehot_mlp', got {encoder_type}"
        
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.encoder_type = encoder_type
        
        if encoder_type == 'embedding':
            # Learnable embedding lookup table
            self.embedding = nn.Embedding(num_actions, embedding_dim)
        else:  # onehot_mlp
            # One-hot to MLP mapping
            self.mlp = nn.Sequential(
                nn.Linear(num_actions, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim)
            )
        
        # Apply weight initialization
        self.apply(initialize_weights)
        
        # Print model statistics
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[ActionEncoder] Number of parameters: {num_params}")
    
    def forward(self, actions: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for ActionEncoder.
        
        Args:
            actions (torch.LongTensor): Action indices of shape (B,) or (B, seq_len)
                                      with values in [0, num_actions-1]
        
        Returns:
            torch.Tensor: Action embeddings of shape (B, embedding_dim) or (B, seq_len, embedding_dim)
        """
        # Validate input range
        if torch.any(actions < 0) or torch.any(actions >= self.num_actions):
            raise ValueError(f"Action indices must be in range [0, {self.num_actions-1}], "
                           f"got min={actions.min().item()}, max={actions.max().item()}")
        
        if self.encoder_type == 'embedding':
            # Direct embedding lookup
            return self.embedding(actions)
        else:  # onehot_mlp
            # Convert to one-hot and pass through MLP
            one_hot = F.one_hot(actions, num_classes=self.num_actions).float()
            return self.mlp(one_hot)
