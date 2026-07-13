import torch
from .symmetric import SymmetricLinearSystem
from .invertible import InvertibleLinearSystem

class DiagonalScalar(SymmetricLinearSystem, InvertibleLinearSystem):
    def __init__(self, diagonal_vector):
        """
        This class implements a diagonal linear system.
        
        parameters:
            diagonal_vector: torch.Tensor or list or omegaconf.ListConfig
                The diagonal elements of the linear system.
        """
        super().__init__()
        
        # Convert to torch.Tensor if needed
        if isinstance(diagonal_vector, (int, float)):
            diagonal_vector = torch.tensor(diagonal_vector)
        elif hasattr(diagonal_vector, '__iter__') and not isinstance(diagonal_vector, torch.Tensor):
            # Handle list/tuple/omegaconf.ListConfig from config
            diagonal_vector = torch.tensor(list(diagonal_vector))
        
        self.diagonal_vector = diagonal_vector
    
    @property
    def is_invertible(self) -> bool:
        """
        Check if this diagonal operator is invertible.
        
        returns:
            bool: True if diagonal vector contains no zeros, False otherwise.
        """
        return not torch.any(self.diagonal_vector == 0)
    
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
        return self.diagonal_vector * x
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate of the linear system.
        
        parameters:
            x: torch.Tensor
                The input tensor to the conjugate of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the conjugate of the linear system to the input tensor.
        """
        return torch.conj(self.diagonal_vector) * x
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear system.
        
        parameters:
            y: torch.Tensor
                The input tensor to the inverse of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the inverse of the linear system to the input tensor.
        raises:
            ValueError: If the diagonal vector contains zeros.
        """
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return y / self.diagonal_vector
    
    def inverse_LinearSystem(self):
        """
        This method returns the inverse linear system.
        
        returns:
            result: DiagonalScalar
                The inverse linear system.
        raises:
            ValueError: If the diagonal vector contains zeros.
        """
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return DiagonalScalar(1 / self.diagonal_vector)
    
    def sqrt_LinearSystem(self):
        """
        This method returns the square root linear system.
        
        returns:
            result: DiagonalScalar
                The square root linear system.
        """
        return DiagonalScalar(torch.sqrt(self.diagonal_vector))
    
    def mat_add(self, other):
        """
        This method implements matrix addition with another linear system.
        
        parameters:
            other: DiagonalScalar or Scalar or Identity
                The other linear system to add.
        returns:
            result: DiagonalScalar
                The sum of the two linear systems.
        """
        from .scalar import Scalar
        from .identity import Identity
        
        if isinstance(other, DiagonalScalar):
            return DiagonalScalar(self.diagonal_vector + other.diagonal_vector)
        elif isinstance(other, Scalar):
            # Scalar + Diagonal = Diagonal with scalar added to each element
            return DiagonalScalar(self.diagonal_vector + other.scalar)
        elif isinstance(other, Identity):
            # Identity + Diagonal = Diagonal with 1 added to each element
            return DiagonalScalar(self.diagonal_vector + 1.0)
        else:
            raise ValueError(f"Addition with {type(other)} not supported for DiagonalScalar")
    
    def mat_sub(self, other):
        """
        This method implements matrix subtraction with another linear system.
        
        parameters:
            other: DiagonalScalar or Scalar or Identity
                The other linear system to subtract.
        returns:
            result: DiagonalScalar
                The difference of the two linear systems.
        """
        from .scalar import Scalar
        from .identity import Identity
        
        if isinstance(other, DiagonalScalar):
            return DiagonalScalar(self.diagonal_vector - other.diagonal_vector)
        elif isinstance(other, Scalar):
            # Diagonal - Scalar = Diagonal with scalar subtracted from each element
            return DiagonalScalar(self.diagonal_vector - other.scalar)
        elif isinstance(other, Identity):
            # Diagonal - Identity = Diagonal with 1 subtracted from each element
            return DiagonalScalar(self.diagonal_vector - 1.0)
        else:
            raise ValueError(f"Subtraction with {type(other)} not supported for DiagonalScalar")
    
    def mat_mul(self, other):
        """
        This method implements matrix multiplication with another linear system.
        
        parameters:
            other: DiagonalScalar or Scalar or Identity
                The other linear system to multiply.
        returns:
            result: DiagonalScalar
                The product of the two linear systems.
        """
        from .scalar import Scalar
        from .identity import Identity
        
        if isinstance(other, DiagonalScalar):
            return DiagonalScalar(self.diagonal_vector * other.diagonal_vector)
        elif isinstance(other, Scalar):
            # Diagonal * Scalar = Diagonal with each element multiplied by scalar
            return DiagonalScalar(self.diagonal_vector * other.scalar)
        elif isinstance(other, Identity):
            # Diagonal * Identity = Diagonal (unchanged)
            return DiagonalScalar(self.diagonal_vector)
        else:
            raise ValueError(f"Multiplication with {type(other)} not supported for DiagonalScalar")

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate transpose of the linear system.
        
        For diagonal operators, conjugate transpose equals conjugate.
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the linear system to the input tensor.
        """
        return self.conjugate(y)

    def transpose_LinearSystem(self):
        """
        This method returns the transpose linear system.
        
        For diagonal operators, transpose equals the operator itself (diagonal matrices are symmetric).
        
        returns:
            result: DiagonalScalar
                The transpose linear system (same as self).
        """
        return DiagonalScalar(self.diagonal_vector)

    def conjugate_LinearSystem(self):
        """
        This method returns the conjugate linear system.
        
        returns:
            result: DiagonalScalar
                The conjugate linear system.
        """
        return DiagonalScalar(torch.conj(self.diagonal_vector))

    def conjugate_transpose_LinearSystem(self):
        """
        This method returns the conjugate transpose linear system.
        
        returns:
            result: DiagonalScalar
                The conjugate transpose linear system.
        """
        return DiagonalScalar(torch.conj(self.diagonal_vector))

    def __matmul__(self, other):
        """
        This method implements the @ operator for matrix multiplication.
        
        parameters:
            other: LinearSystem or torch.Tensor
                The other linear system to multiply, or a tensor.
        returns:
            result: LinearSystem or torch.Tensor
                The product of the two linear systems, or the result of applying to tensor.
        """
        return self.mat_mul(other)

    def logdet(self):
        """
        This method returns the log determinant of the linear system.
        
        returns:
            result: torch.Tensor
                The log determinant.
        """
        return torch.sum(torch.log(torch.abs(self.diagonal_vector)))

    def det(self):
        """
        This method returns the determinant of the linear system.
        
        returns:
            result: torch.Tensor
                The determinant.
        """
        return torch.prod(self.diagonal_vector)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear system.
        
        For diagonal operators, transpose equals forward.
        
        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the transpose of the linear system to the input tensor.
        """
        return self.forward(y)
