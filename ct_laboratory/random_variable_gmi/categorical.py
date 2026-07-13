import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .core import RandomVariable


class CategoricalRandomVariable(RandomVariable):
    """
    Categorical random variable defined by a vector of logits.
    The categorical probability mass function is the softmax of the logits.
    It is a discrete distribution interpreted as a sum of weighted dirac delta impulses,
    centered at integers 0 through K-1 where K is the number of classes.
    """
    
    def __init__(self, logits: torch.Tensor, codebook: Optional[Dict[int, str]] = None):
        """
        Initialize categorical random variable.
        
        Args:
            logits: Tensor of shape (num_classes,) containing the logits
            codebook: Optional dictionary mapping class indices to names
        """
        super().__init__()
        self.logits = nn.Parameter(logits.clone())
        self.codebook = codebook or {}
        self.num_classes = len(logits)
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for one-hot encoded input.
        
        Args:
            x: One-hot encoded tensor of shape (batch_size, num_classes)
        
        Returns:
            Log probabilities of shape (batch_size,)
        """
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(self.logits, dim=0)
        
        # For one-hot vectors, we can compute log_prob as dot product
        return torch.sum(x * log_probs, dim=-1)
    
    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample from the categorical distribution.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            One-hot encoded samples of shape (batch_size, num_classes)
        """
        # Sample class indices
        probs = F.softmax(self.logits, dim=0)
        indices = torch.multinomial(probs, batch_size, replacement=True)
        
        # Convert to one-hot
        one_hot = torch.zeros(batch_size, self.num_classes, device=self.logits.device)
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        
        return one_hot
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from codebook or return default."""
        return self.codebook.get(class_idx, f"Class {class_idx}")


class ConditionalCategoricalRandomVariable(RandomVariable):
    """
    Conditional categorical random variable with a deterministic function (nn.Module)
    mapping conditional information (e.g., images) to categorical logit vectors.
    """
    
    def __init__(self, logit_function: nn.Module, codebook: Optional[Dict[int, str]] = None):
        """
        Initialize conditional categorical random variable.
        
        Args:
            logit_function: Neural network that maps inputs to logits
            codebook: Optional dictionary mapping class indices to names
        """
        super().__init__()
        self.logit_function = logit_function
        self.codebook = codebook or {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the logit function."""
        return self.logit_function(x)
    
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability for conditional input.
        
        Args:
            x: Conditional input (e.g., images) of shape (batch_size, ...)
            y: One-hot encoded target of shape (batch_size, num_classes)
        
        Returns:
            Log probabilities of shape (batch_size,)
        """
        logits = self.logit_function(x)
        
        # Use log_softmax for numerical stability and direct computation
        log_probs = F.log_softmax(logits, dim=1)
        
        # For one-hot vectors, we can compute log_prob as element-wise multiplication and sum
        return torch.sum(y * log_probs, dim=1)
    
    def train_loss_closure(self, batch):
        """Training loss closure - negative log probability."""
        self.train()
        if len(batch) == 2:
            x, y = batch
            return -self.log_prob(x, y).mean()
        else:
            raise ValueError("Batch should contain (x, y) pairs for conditional model")
    
    def eval_loss_closure(self, batch):
        """Evaluation loss closure - negative log probability."""
        self.eval()
        with torch.no_grad():
            if len(batch) == 2:
                x, y = batch
                return -self.log_prob(x, y).mean()
            else:
                raise ValueError("Batch should contain (x, y) pairs for conditional model")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (return logits).
        
        Args:
            x: Input tensor
            
        Returns:
            Logits tensor
        """
        return self.logit_function(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        logits = self.logit_function(x)
        return F.softmax(logits, dim=1)
    
    def predict_classes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class indices.
        
        Args:
            x: Input tensor
            
        Returns:
            Class indices tensor
        """
        logits = self.logit_function(x)
        return torch.argmax(logits, dim=1)
    
    def get_class_name(self, class_idx: int) -> str:
        """Get class name from codebook or return default."""
        return self.codebook.get(class_idx, f"Class {class_idx}")