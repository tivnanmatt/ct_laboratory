import torch
from .base import LinearSystem

class ConjugateLinearSystem(LinearSystem):
    def __init__(self, base_matrix_operator):
        """
        This class represents the conjugate of another linear system.
        
        parameters:
            base_matrix_operator: LinearSystem
                The linear system to which the conjugate should be applied.
        """
        super().__init__()
        self.base_matrix_operator = base_matrix_operator
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(y)
    
    def conjugate_LinearSystem(self):
        return ConjugateLinearSystem(self.base_matrix_operator)
    
    def __repr__(self):
        return f"ConjugateLinearSystem({self.base_matrix_operator})" 