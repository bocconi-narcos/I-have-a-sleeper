import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from src.utils.weight_init import initialize_weights


class TransitionModel(nn.Module):
    """
    TransitionModel for discrete-MDP world-model.
    
    Predicts the distribution over next state embeddings x_{t+1} given (x_t, e_t),
    approximating P(s_{t+1} | s_t, a_t) ≈ T(x_t, e_t).
    
    Architecture follows Eq. (3) from the paper and is trained with negative log-likelihood
    loss of Eq. (5).
    
    Args:
        state_dim (int): Dimension of state embeddings x_t (should match latent_dim from StateEncoder/StateDecoder)
        action_dim (int): Dimension of action embeddings e_t (should match embedding_dim from ActionEncoder)
        hidden_dim (int): Hidden layer dimension. Defaults to 256.
        dropout (float): Dropout rate. Defaults to 0.0.
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        input_dim = state_dim + action_dim
        
        # Two hidden layers with ReLU activation as specified
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers for Gaussian distribution parameters
        self.mu_head = nn.Linear(hidden_dim, state_dim)
        self.logvar_head = nn.Linear(hidden_dim, state_dim)
        
        # Apply weight initialization
        self.apply(initialize_weights)
        
        # Print model statistics
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[TransitionModel] Number of parameters: {num_params}")
    
    def forward(self, x_t: torch.Tensor, e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the TransitionModel.
        
        Args:
            x_t (torch.Tensor): Current state embeddings of shape (B, state_dim)
            e_t (torch.Tensor): Action embeddings of shape (B, action_dim)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: mu and logvar of q(x_{t+1} | x_t, e_t)
                - mu: shape (B, state_dim)
                - logvar: shape (B, state_dim)
        """
        # Validate input dimensions
        if x_t.shape[-1] != self.state_dim:
            raise ValueError(f"Expected state embedding dimension {self.state_dim}, "
                           f"got {x_t.shape[-1]}")
        if e_t.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action embedding dimension {self.action_dim}, "
                           f"got {e_t.shape[-1]}")
        if x_t.shape[0] != e_t.shape[0]:
            raise ValueError(f"Batch size mismatch: state {x_t.shape[0]}, action {e_t.shape[0]}")
        
        # Concatenate state and action embeddings
        x_concat = torch.cat([x_t, e_t], dim=-1)
        
        # Pass through network
        hidden = self.network(x_concat)
        
        # Get Gaussian parameters
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        
        return mu, logvar
    
    def sample(self, x_t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """
        Sample next state embeddings from the predicted distribution.
        
        Args:
            x_t (torch.Tensor): Current state embeddings of shape (B, state_dim)
            e_t (torch.Tensor): Action embeddings of shape (B, action_dim)
            
        Returns:
            torch.Tensor: Sampled next state embeddings x_{t+1} of shape (B, state_dim)
                         Ready for state_decoder input - compatible with ARC_StateDecoder.forward(z).
        """
        mu, logvar = self.forward(x_t, e_t)
        
        # Reparameterization trick: x = mu + eps * std
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        x_next = mu + eps * std
        
        return x_next
    
    def log_prob(self, x_t: torch.Tensor, e_t: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of next state under the predicted distribution.
        Useful for computing negative log-likelihood loss (Eq. 5).
        
        Args:
            x_t (torch.Tensor): Current state embeddings of shape (B, state_dim)
            e_t (torch.Tensor): Action embeddings of shape (B, action_dim)
            x_next (torch.Tensor): Next state embeddings of shape (B, state_dim)
            
        Returns:
            torch.Tensor: Log probabilities of shape (B,)
        """
        mu, logvar = self.forward(x_t, e_t)
        
        # Compute log probability under multivariate Gaussian
        # log N(x; mu, sigma^2) = -0.5 * (log(2π) + log(sigma^2) + (x-mu)^2/sigma^2)
        var = torch.exp(logvar)
        log_prob = -0.5 * (torch.log(2 * torch.pi * var) + (x_next - mu) ** 2 / var)
        
        # Sum over dimensions to get log probability for each sample
        log_prob = log_prob.sum(dim=-1)
        
        return log_prob
    
    def nll_loss(self, x_t: torch.Tensor, e_t: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss (Eq. 5).
        
        Args:
            x_t (torch.Tensor): Current state embeddings of shape (B, state_dim)
            e_t (torch.Tensor): Action embeddings of shape (B, action_dim)
            x_next (torch.Tensor): Next state embeddings of shape (B, state_dim)
            
        Returns:
            torch.Tensor: Negative log-likelihood loss (scalar)
        """
        log_prob = self.log_prob(x_t, e_t, x_next)
        return -log_prob.mean()
