import torch
from abc import ABC, abstractmethod
from .base import LinearSystem

class RealLinearSystem(LinearSystem, ABC):
    def __init__(self):
        super().__init__()

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        # For real operators, conjugate is just forward
        return self.forward(x)

    def conjugate_LinearSystem(self):
        return self

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.transpose(y)

    def conjugate_transpose_LinearSystem(self):
        return self 