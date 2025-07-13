import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights


class ARC_ActionDecoder(nn.Module):
    """
    ActionDecoder (f): JSAE-style decoder using nearest-neighbor lookup in embedding space.
    
    Rather than using an MLP to output logits, this decoder finds the closest action embedding
    in Euclidean distance and returns the corresponding action index.
    
    - Input: embedding E_t ∈ ℝᵈ
    - Output: action indices (via nearest-neighbor lookup)
    
    Args:
        action_encoder (nn.Module): The action encoder containing the embedding table
        embedding_dim (int): Dimension of the action embeddings
        num_actions (int): Number of discrete actions
    """
    
    def __init__(self,
                 action_encoder: nn.Module,
                 embedding_dim: int,
                 num_actions: int):
        super().__init__()
        
        self.action_encoder = action_encoder
        self.embedding_dim = embedding_dim
        self.num_actions = num_actions
        
        # Verify that the action encoder uses embedding approach
        if not hasattr(action_encoder, 'embedding'):
            raise ValueError("Action encoder must have an 'embedding' attribute (nn.Embedding table)")
        
        print(f"[ActionDecoder] Using nearest-neighbor lookup with {num_actions} actions")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using nearest-neighbor lookup.
        
        Args:
            embeddings (torch.Tensor): Action embeddings of shape (B, embedding_dim) or 
                                     (B, seq_len, embedding_dim)
        
        Returns:
            torch.Tensor: Action indices of shape (B,) or (B, seq_len)
        """
        # Validate input dimension
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, "
                           f"got {embeddings.shape[-1]}")
        
        # Get all action embeddings from the encoder's embedding table
        # Shape: (num_actions, embedding_dim)
        action_embeddings = self.action_encoder.embedding.weight
        
        # Handle different input shapes
        original_shape = embeddings.shape
        if embeddings.dim() == 3:  # (B, seq_len, embedding_dim)
            batch_size, seq_len, emb_dim = embeddings.shape
            embeddings = embeddings.view(-1, emb_dim)  # (B*seq_len, embedding_dim)
        elif embeddings.dim() == 2:  # (B, embedding_dim)
            batch_size = embeddings.shape[0]
            seq_len = None
        else:
            raise ValueError(f"Expected 2D or 3D input, got {embeddings.dim()}D")
        
        # Compute distances between embeddings and all action embeddings
        # embeddings: (B*seq_len, embedding_dim) or (B, embedding_dim)
        # action_embeddings: (num_actions, embedding_dim)
        # distances: (B*seq_len, num_actions) or (B, num_actions)
        distances = torch.cdist(embeddings, action_embeddings, p=2)
        
        # Find nearest action for each embedding
        action_indices = torch.argmin(distances, dim=-1)  # (B*seq_len,) or (B,)
        
        # Reshape back to original shape
        if seq_len is not None:
            action_indices = action_indices.view(batch_size, seq_len)
        
        return action_indices
    
    def get_action_logits(self, embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get action logits based on negative distances (for loss computation).
        
        Args:
            embeddings (torch.Tensor): Action embeddings of shape (B, embedding_dim) or 
                                     (B, seq_len, embedding_dim)
            temperature (float): Temperature for softmax (lower = more confident)
        
        Returns:
            torch.Tensor: Action logits of shape (B, num_actions) or (B, seq_len, num_actions)
        """
        # Validate input dimension
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, "
                           f"got {embeddings.shape[-1]}")
        
        # Get all action embeddings from the encoder's embedding table
        action_embeddings = self.action_encoder.embedding.weight
        
        # Handle different input shapes
        original_shape = embeddings.shape
        if embeddings.dim() == 3:  # (B, seq_len, embedding_dim)
            batch_size, seq_len, emb_dim = embeddings.shape
            embeddings = embeddings.view(-1, emb_dim)  # (B*seq_len, embedding_dim)
        elif embeddings.dim() == 2:  # (B, embedding_dim)
            batch_size = embeddings.shape[0]
            seq_len = None
        else:
            raise ValueError(f"Expected 2D or 3D input, got {embeddings.dim()}D")
        
        # Compute distances and convert to logits
        distances = torch.cdist(embeddings, action_embeddings, p=2)
        logits = -distances / temperature  # Negative distances as logits
        
        # Reshape back to original shape
        if seq_len is not None:
            logits = logits.view(batch_size, seq_len, self.num_actions)
        
        return logits
