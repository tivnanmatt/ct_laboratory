import torch
from abc import ABC, abstractmethod
from .square import SquareLinearSystem

class InvertibleLinearSystem(SquareLinearSystem, ABC):
    def __init__(self):
        super().__init__()

    @property
    def is_invertible(self) -> bool:
        """
        Check if this linear system is invertible.
        
        Default implementation returns True for invertible operators.
        Subclasses can override this to provide specific logic.
        
        returns:
            bool: True by default, can be overridden by subclasses.
        """
        return True

    @abstractmethod
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear system.
        
        parameters:
            y: torch.Tensor
                The input tensor to the inverse of the linear system.
        returns:
            x: torch.Tensor
                The result of applying the inverse of the linear system to the input tensor.
        """
        pass

    def inverse_LinearSystem(self):
        from .inverse import InverseLinearSystem
        return InverseLinearSystem(self) 