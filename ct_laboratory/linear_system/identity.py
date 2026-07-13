import torch
from .scalar import Scalar
from .real import RealLinearSystem
from .hermitian import HermitianLinearSystem
from .unitary import UnitaryLinearSystem

class Identity(Scalar, RealLinearSystem, HermitianLinearSystem, UnitaryLinearSystem):
    def __init__(self):
        """
        Identity linear system (the identity matrix).
        """
        super().__init__(scalar=1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear system.
        
        parameters:
            x: torch.Tensor
                The input tensor to the linear system.
        returns:
            result: torch.Tensor
                The result of applying the linear system to the input tensor.
        """
        return x
    

    
    def mat_add(self, other):
        """
        This method implements matrix addition with another linear system.
        
        parameters:
            other: Identity or Scalar or DiagonalScalar
                The other linear system to add.
        returns:
            result: Scalar or DiagonalScalar
                The sum of the two linear systems.
        """
        from .scalar import Scalar
        from .diagonal import DiagonalScalar
        
        if isinstance(other, Identity):
            return Scalar(2.0)
        elif isinstance(other, Scalar):
            return Scalar(1.0 + other.scalar)
        elif isinstance(other, DiagonalScalar):
            # Identity + Diagonal = Diagonal with 1 added to each element
            return DiagonalScalar(1.0 + other.diagonal_vector)
        else:
            raise ValueError(f"Addition with {type(other)} not supported for Identity")
    
    def mat_sub(self, other):
        """
        This method implements matrix subtraction with another linear system.
        
        parameters:
            other: Identity or Scalar or DiagonalScalar
                The other linear system to subtract.
        returns:
            result: Scalar or DiagonalScalar
                The difference of the two linear systems.
        """
        from .scalar import Scalar
        from .diagonal import DiagonalScalar
        
        if isinstance(other, Identity):
            return Scalar(0.0)
        elif isinstance(other, Scalar):
            return Scalar(1.0 - other.scalar)
        elif isinstance(other, DiagonalScalar):
            # Identity - Diagonal = Diagonal with 1 subtracted from each element
            return DiagonalScalar(1.0 - other.diagonal_vector)
        else:
            raise ValueError(f"Subtraction with {type(other)} not supported for Identity")
    
    def mat_mul(self, other):
        """
        This method implements matrix multiplication with another linear system.
        
        parameters:
            other: Identity or Scalar or DiagonalScalar or torch.Tensor
                The other linear system to multiply, or a tensor.
        returns:
            result: Identity or Scalar or DiagonalScalar or torch.Tensor
                The product of the two linear systems, or the result of applying to tensor.
        """
        from .scalar import Scalar
        from .diagonal import DiagonalScalar
        
        if isinstance(other, torch.Tensor):
            return other
        elif isinstance(other, Identity):
            return Identity()
        elif isinstance(other, Scalar):
            return Scalar(other.scalar)
        elif isinstance(other, DiagonalScalar):
            # Identity * Diagonal = Diagonal (unchanged)
            return DiagonalScalar(other.diagonal_vector)
        else:
            # For other types, multiplication by identity returns the other operator
            return other
    
    def logdet(self):
        """
        This method returns the log determinant of the linear system.
        
        returns:
            result: torch.Tensor
                The log determinant (which is 0 for identity).
        """
        return torch.tensor(0.0)

    def inverse_LinearSystem(self):
        """
        This method returns the inverse linear system.
        
        returns:
            result: Identity
                The inverse linear system (which is the identity itself).
        """
        return Identity()

    def sqrt_LinearSystem(self):
        """
        This method returns the square root linear system.
        
        returns:
            result: Identity
                The square root linear system (which is the identity itself).
        """
        return Identity()

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear system.
        
        For identity operators, transpose equals forward.
        
        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the transpose of the linear system to the input tensor.
        """
        return self.forward(y)

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate transpose of the linear system.
        
        For real identity operators, conjugate transpose equals forward.
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the linear system to the input tensor.
        """
        return self.forward(y) 