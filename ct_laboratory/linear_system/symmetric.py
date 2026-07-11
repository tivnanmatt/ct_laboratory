import torch
from abc import ABC, abstractmethod
from .square import SquareLinearSystem

class SymmetricLinearSystem(SquareLinearSystem, ABC):
    def __init__(self):
        super().__init__()

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        # For symmetric operators, transpose = forward
        return self.forward(y)

    def transpose_LinearSystem(self):
        return self
    
 