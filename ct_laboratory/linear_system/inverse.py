"""
Inverse Linear System

This module provides the InverseLinearSystem class, which represents the inverse
of an invertible linear system.
"""

import torch
from typing import Optional

from .invertible import InvertibleLinearSystem
from .base import LinearSystem


class InverseLinearSystem(InvertibleLinearSystem):
    """
    Linear system that represents the inverse of another invertible linear system.
    
    This class wraps an invertible linear system and provides its inverse
    as the forward operation. The inverse of this operator is the original operator.
    
    Parameters:
        base_matrix_operator: InvertibleLinearSystem
            The invertible linear system to invert.
    """
    
    def __init__(self, base_matrix_operator: InvertibleLinearSystem):
        """
        Initialize the inverse linear system.
        
        Parameters:
            base_matrix_operator: InvertibleLinearSystem
                The invertible linear system to invert.
        """
        assert isinstance(base_matrix_operator, InvertibleLinearSystem), \
            "The input linear system should be an InvertibleLinearSystem object."
        
        super().__init__()
        self.base_matrix_operator = base_matrix_operator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the inverse of the base operator to the input.
        
        Parameters:
            x: torch.Tensor
                Input tensor.
                
        Returns:
            torch.Tensor: Result of applying the inverse operator.
        """
        return self.base_matrix_operator.inverse(x)
    
    def inverse_LinearSystem(self) -> LinearSystem:
        """
        Return the original base operator as the inverse of this operator.
        
        Returns:
            LinearSystem: The original base operator.
        """
        return self.base_matrix_operator.forward_LinearSystem()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the original base operator to the input.
        
        Parameters:
            y: torch.Tensor
                Input tensor.
                
        Returns:
            torch.Tensor: Result of applying the original operator.
        """
        return self.base_matrix_operator.forward(y) 