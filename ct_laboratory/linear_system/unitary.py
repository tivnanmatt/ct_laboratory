import torch
from .invertible import InvertibleLinearSystem

class UnitaryLinearSystem(InvertibleLinearSystem):
    def __init__(self):
        super().__init__()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # For unitary operators, inverse = conjugate_transpose
        return self.conjugate_transpose(y)

    def inverse_LinearSystem(self):
        from .conjugate_transpose import ConjugateTransposeLinearSystem
        return ConjugateTransposeLinearSystem(self) 