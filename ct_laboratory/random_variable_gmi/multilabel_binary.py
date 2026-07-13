import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .core import RandomVariable


class MultilabelBinaryRandomVariable(RandomVariable):
    """
    A multi-label binary random variable for multi-label classification.
    Each label is independent and can be 0 or 1.
    
    This is used for tasks like ChestMNIST where each sample can have
    multiple conditions simultaneously (multi-hot encoding).
    """
    
    def __init__(self, logits: torch.Tensor, codebook: Optional[Dict[int, str]] = None):
        """
        Args:
            logits: Tensor of shape (num_labels,) containing the logits for each binary label
            codebook: Dictionary mapping label indices to human-readable names
        """
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.codebook = codebook or {}
        self.num_labels = len(logits)
    
    def log_prob(self, multi_hot_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of multi-hot labels.
        
        Args:
            multi_hot_labels: Multi-hot encoded labels [batch_size, num_labels]
                             Each element is 0 or 1 indicating absence/presence
        
        Returns:
            log_prob: Log probability tensor [batch_size]
        """
        # Expand logits to match batch size
        batch_size = multi_hot_labels.shape[0]
        logits = self.logits.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_labels]c
        
        # Ensure multi_hot_labels is float for BCE computation
        if multi_hot_labels.dtype != torch.float32:
            multi_hot_labels = multi_hot_labels.float()
        
        # Compute log probabilities using binary cross entropy formulation
        log_probs_per_label = -F.binary_cross_entropy_with_logits(
            logits, multi_hot_labels, reduction='none'
        )  # [batch_size, num_labels]
        
        # Sum across all labels to get total log probability
        log_prob = log_probs_per_label.sum(dim=1)  # [batch_size]
        
        return log_prob


class ConditionalMultilabelBinaryRandomVariable(RandomVariable):
    """
    Conditional multi-label binary random variable with a deterministic function (nn.Module)
    mapping conditional information (e.g., images) to multi-label binary logit vectors.
    """
    
    def __init__(self, logit_function: nn.Module, codebook: Optional[Dict[int, str]] = None):
        """
        Initialize conditional multi-label binary random variable.
        
        Args:
            logit_function: Neural network that maps inputs to logits for binary labels
            codebook: Optional dictionary mapping label indices to names
        """
        super().__init__()
        self.logit_function = logit_function
        self.codebook = codebook or {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the logit function."""
        return self.logit_function(x)
    
    def log_prob(self, x: torch.Tensor, multi_hot_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for conditional input.
        
        Args:
            x: Conditional input (e.g., images) of shape (batch_size, ...)
            multi_hot_labels: Multi-hot encoded labels [batch_size, num_labels]
                             Each element is 0 or 1 indicating absence/presence
        
        Returns:
            log_prob: Log probability tensor [batch_size]
        """
        # Get logits from the network
        logits = self.logit_function(x)  # [batch_size, num_labels]
        
        # Ensure multi_hot_labels is float for BCE computation
        if multi_hot_labels.dtype != torch.float32:
            multi_hot_labels = multi_hot_labels.float()
        
        # Compute log probabilities using binary cross entropy formulation
        log_probs_per_label = -F.binary_cross_entropy_with_logits(
            logits, multi_hot_labels, reduction='none'
        )  # [batch_size, num_labels]
        
        # Sum across all labels to get total log probability
        log_prob = log_probs_per_label.sum(dim=1)  # [batch_size]
        
        return log_prob
    
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict probabilities for each label independently.
        
        Args:
            x: Input tensor [batch_size, ...]
            
        Returns:
            probs: Sigmoid probabilities [batch_size, num_labels]
        """
        self.eval()
        with torch.no_grad():
            logits = self.logit_function(x)
            probs = torch.sigmoid(logits)
        return probs
    
    def predict_binary(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary labels using a threshold.
        
        Args:
            x: Input tensor [batch_size, ...]
            threshold: Threshold for binary classification (default: 0.5)
            
        Returns:
            predictions: Binary predictions [batch_size, num_labels]
        """
        probs = self.predict_probabilities(x)
        return (probs > threshold).float()
    
    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample binary labels from the predicted distribution.
        
        Args:
            x: Input tensor [batch_size, ...]
            temperature: Temperature for sampling (default: 1.0)
            
        Returns:
            samples: Binary samples [batch_size, num_labels]
        """
        self.eval()
        with torch.no_grad():
            logits = self.logit_function(x) / temperature
            probs = torch.sigmoid(logits)
            samples = torch.bernoulli(probs)
        return samples