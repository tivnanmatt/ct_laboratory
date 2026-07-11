import torch
from abc import ABC, abstractmethod
from .square import SquareLinearSystem

class HermitianLinearSystem(SquareLinearSystem, ABC):
    def __init__(self):
        super().__init__()

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        # For hermitian operators, conjugate_transpose = forward
        return self.forward(y)

    def conjugate_transpose_LinearSystem(self):
        return self 