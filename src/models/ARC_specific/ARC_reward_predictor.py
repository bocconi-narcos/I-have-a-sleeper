import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RewardPredictor(nn.Module):
    """
    RewardPredictor module that takes three sequences of state embeddings
    (current, next, goal) and predicts a scalar reward using a Transformer encoder.
    
    Uses type embeddings to distinguish between sequence types and pooling
    instead of a special CLS token.
    """
    
    def __init__(self,
                 d_model: int = 128,
                 n_heads: int = 8,
                 num_layers: int = 4,
                 dim_ff: int = 512,
                 dropout: float = 0.1,
                 use_positional_encoding: bool = True,
                 pooling_method: str = "mean"):
        """
        Initialize the RewardPredictor.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_ff: Dimension of feed-forward network
            dropout: Dropout probability
            use_positional_encoding: Whether to use positional encoding
            pooling_method: Pooling method ("mean" or "max")
        """
        super().__init__()
        
        self.d_model = d_model
        self.pooling_method = pooling_method
        self.use_positional_encoding = use_positional_encoding
        
        # Type embeddings for segment IDs: {0: current, 1: next, 2: goal}
        self.type_embeddings = nn.Embedding(3, d_model)
        
        # Positional encoding (optional)
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=5000)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Reward prediction head (two-layer MLP)
        self.ff1 = nn.Linear(d_model, dim_ff // 2)
        self.ff2 = nn.Linear(dim_ff // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize type embeddings
        nn.init.normal_(self.type_embeddings.weight, std=0.02)
        
        # Initialize MLP layers
        nn.init.xavier_uniform_(self.ff1.weight)
        nn.init.xavier_uniform_(self.ff2.weight)
        nn.init.zeros_(self.ff1.bias)
        nn.init.zeros_(self.ff2.bias)
    
    def forward(self, 
                z_t: torch.Tensor, 
                z_tp1: torch.Tensor, 
                z_goal: torch.Tensor,
                r_true: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the RewardPredictor.
        
        Args:
            z_t: Current state embeddings (B, L, D)
            z_tp1: Next state embeddings (B, L, D) 
            z_goal: Goal state embeddings (B, L, D)
            r_true: True rewards for loss calculation (B,) [optional]
            mask: Attention mask (B, 3L) [optional]
        
        Returns:
            r_pred: Predicted rewards (B, 1)
            loss: MSE loss if r_true is provided, else None
        """
        batch_size, seq_len, d_model = z_t.shape
        
        # Concatenate sequences along sequence dimension
        x = torch.cat([z_t, z_tp1, z_goal], dim=1)  # (B, 3L, D)
        
        # Create type embeddings for each segment
        device = x.device
        type_ids = torch.cat([
            torch.zeros(seq_len, dtype=torch.long, device=device),      # current
            torch.ones(seq_len, dtype=torch.long, device=device),       # next
            torch.full((seq_len,), 2, dtype=torch.long, device=device)  # goal
        ])  # (3L,)
        
        # Expand type_ids for batch dimension and get embeddings
        type_ids = type_ids.unsqueeze(0).expand(batch_size, -1)  # (B, 3L)
        type_embeds = self.type_embeddings(type_ids)  # (B, 3L, D)
        
        # Add type embeddings to input
        x = x + type_embeds
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Create attention mask if provided
        # mask should be True for positions to attend to, False to ignore
        if mask is not None:
            # Convert boolean mask to float mask for transformer
            # True -> 0.0 (attend), False -> -inf (ignore)
            attn_mask = mask.float()
            attn_mask = attn_mask.masked_fill(mask == False, float('-inf'))
            attn_mask = attn_mask.masked_fill(mask == True, 0.0)
        else:
            attn_mask = None
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # (B, 3L, D)
        
        # Pool the sequence to get a single vector per example
        if self.pooling_method == "mean":
            if mask is not None:
                # Masked mean pooling
                mask_expanded = mask.unsqueeze(-1).float()  # (B, 3L, 1)
                x_masked = x * mask_expanded
                h = x_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
            else:
                h = x.mean(dim=1)  # (B, D)
        elif self.pooling_method == "max":
            if mask is not None:
                # Masked max pooling
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                h, _ = x_masked.max(dim=1)  # (B, D)
            else:
                h, _ = x.max(dim=1)  # (B, D)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        # Pass through reward prediction head
        hidden = F.relu(self.ff1(h))  # (B, dim_ff//2)
        hidden = self.dropout(hidden)
        r_pred = self.ff2(hidden)  # (B, 1)
        
        # Calculate loss if true rewards are provided
        loss = None
        if r_true is not None:
            loss = F.mse_loss(r_pred.squeeze(-1), r_true)
        
        return r_pred, loss


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor (B, L, D)
        
        Returns:
            x with positional encoding added (B, L, D)
        """
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :, :].transpose(0, 1)  # (B, L, D)
        return self.dropout(x)


