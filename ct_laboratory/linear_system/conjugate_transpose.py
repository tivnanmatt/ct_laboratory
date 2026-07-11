import torch
from .base import LinearSystem

class ConjugateTransposeLinearSystem(LinearSystem):
    def __init__(self, base_matrix_operator):
        """
        This class represents the conjugate transpose of another linear system.
        
        parameters:
            base_matrix_operator: LinearSystem
                The linear system to which the conjugate transpose should be applied.
        """
        super().__init__()
        self.base_matrix_operator = base_matrix_operator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y) 