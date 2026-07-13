import torch
from .base import LinearSystem

class TransposeLinearSystem(LinearSystem):
    def __init__(self, base_matrix_operator):
        """
        This class represents the transpose of another linear system.
        
        parameters:
            base_matrix_operator: LinearSystem
                The linear system to which the transpose should be applied.
        """
        super().__init__()
        self.base_matrix_operator = base_matrix_operator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y) 