import torch
from .diagonal import DiagonalScalar
from .invertible import InvertibleLinearSystem

class Scalar(DiagonalScalar, InvertibleLinearSystem):
    def __init__(self, scalar):
        """
        This class implements a scalar linear system (scalar multiple of identity).
        
        parameters:
            scalar: float or torch.Tensor or list or omegaconf.ListConfig
                The scalar to multiply the input tensor with.
        """
        if isinstance(scalar, (int, float, complex)):
            scalar = torch.tensor(scalar)
        elif hasattr(scalar, '__iter__') and not isinstance(scalar, torch.Tensor):
            # Handle list/tuple/omegaconf.ListConfig from config
            scalar = torch.tensor(list(scalar))
        
        # For scalar operator, we need to handle broadcasting properly
        # The diagonal vector will be the scalar value
        self.scalar = scalar
        super().__init__(scalar)
    
    @property
    def is_invertible(self) -> bool:
        """
        Check if this scalar operator is invertible.
        
        returns:
            bool: True if scalar is non-zero, False otherwise.
        """
        return not torch.any(self.scalar == 0)
    
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
        # Handle broadcasting for scalar multiplication
        if isinstance(self.scalar, torch.Tensor) and self.scalar.dim() > 0:
            for i, shape in enumerate(self.scalar.shape):
                assert x.shape[i] == shape or self.scalar.shape[i] == 1, f"Shape mismatch at dimension {i}: x.shape[{i}]={x.shape[i]}, scalar.shape[{i}]={self.scalar.shape[i]}"
            
            target_shape = list(self.scalar.shape)
            for i in range(len(x.shape) - len(self.scalar.shape)):
                target_shape.append(1)
            
            return self.scalar.reshape(target_shape).to(x.device) * x
        else:  # If scalar is a 0-dimensional tensor (scalar)
            return self.scalar.to(x.device) * x
    
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
        # Handle broadcasting for scalar multiplication
        if isinstance(self.scalar, torch.Tensor) and self.scalar.dim() > 0:
            for i, shape in enumerate(self.scalar.shape):
                assert x.shape[i] == shape or self.scalar.shape[i] == 1, f"Shape mismatch at dimension {i}: x.shape[{i}]={x.shape[i]}, scalar.shape[{i}]={self.scalar.shape[i]}"
            
            target_shape = list(self.scalar.shape)
            for i in range(len(x.shape) - len(self.scalar.shape)):
                target_shape.append(1)
            
            return torch.conj(self.scalar).reshape(target_shape).to(x.device) * x
        else:  # If scalar is a 0-dimensional tensor (scalar)
            return torch.conj(self.scalar).to(x.device) * x
    
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
            ValueError: If the scalar is zero.
        """
        if torch.any(self.scalar == 0):
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return y / self.scalar
    
    def inv_LinearSystem(self):
        """
        This method returns the inverse linear system.
        
        returns:
            result: Scalar
                The inverse linear system.
        raises:
            ValueError: If the scalar is zero.
        """
        if torch.any(self.scalar == 0):
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return Scalar(1 / self.scalar)
    
    def inverse_LinearSystem(self):
        """
        This method returns the inverse linear system.
        
        returns:
            result: Scalar
                The inverse linear system.
        raises:
            ValueError: If the scalar is zero.
        """
        if torch.any(self.scalar == 0):
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return Scalar(1 / self.scalar)
    
    def sqrt_LinearSystem(self):
        """
        This method returns the square root linear system.
        
        returns:
            result: Scalar
                The square root linear system.
        """
        return Scalar(torch.sqrt(self.scalar))
    
    def mat_add(self, other):
        """
        This method implements matrix addition with another linear system.
        
        parameters:
            other: Scalar or DiagonalScalar or Identity
                The other linear system to add.
        returns:
            result: Scalar or DiagonalScalar
                The sum of the two linear systems.
        """
        from .diagonal import DiagonalScalar
        from .identity import Identity
        
        if isinstance(other, Scalar):
            return Scalar(self.scalar + other.scalar)
        elif isinstance(other, DiagonalScalar):
            # Scalar + Diagonal = Diagonal with scalar added to each element
            return DiagonalScalar(self.scalar + other.diagonal_vector)
        elif isinstance(other, Identity):
            # Scalar + Identity = Scalar with 1 added
            return Scalar(self.scalar + 1.0)
        else:
            raise ValueError(f"Addition with {type(other)} not supported for Scalar")
    
    def mat_sub(self, other):
        """
        This method implements matrix subtraction with another linear system.
        
        parameters:
            other: Scalar or DiagonalScalar or Identity
                The other linear system to subtract.
        returns:
            result: Scalar or DiagonalScalar
                The difference of the two linear systems.
        """
        from .diagonal import DiagonalScalar
        from .identity import Identity
        
        if isinstance(other, Scalar):
            return Scalar(self.scalar - other.scalar)
        elif isinstance(other, DiagonalScalar):
            # Scalar - Diagonal = Diagonal with scalar subtracted from each element
            return DiagonalScalar(self.scalar - other.diagonal_vector)
        elif isinstance(other, Identity):
            # Scalar - Identity = Scalar with 1 subtracted
            return Scalar(self.scalar - 1.0)
        else:
            raise ValueError(f"Subtraction with {type(other)} not supported for Scalar")
    
    def mat_mul(self, other):
        """
        This method implements matrix multiplication with another linear system.
        
        parameters:
            other: Scalar or DiagonalScalar or Identity or torch.Tensor
                The other linear system to multiply, or a tensor.
        returns:
            result: Scalar or DiagonalScalar or torch.Tensor
                The product of the two linear systems, or the result of applying to tensor.
        """
        from .diagonal import DiagonalScalar
        from .identity import Identity
        
        if isinstance(other, torch.Tensor):
            return self.forward(other)
        elif isinstance(other, Scalar):
            return Scalar(self.scalar * other.scalar)
        elif isinstance(other, DiagonalScalar):
            # Scalar * Diagonal = Diagonal with each element multiplied by scalar
            return DiagonalScalar(self.scalar * other.diagonal_vector)
        elif isinstance(other, Identity):
            # Scalar * Identity = Scalar (unchanged)
            return Scalar(self.scalar)
        else:
            raise ValueError(f"Multiplication with {type(other)} not supported for Scalar")
    
    def logdet(self):
        """
        This method returns the log determinant of the linear system.
        
        returns:
            result: torch.Tensor
                The log determinant.
        """
        return torch.log(torch.abs(self.scalar))
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear system.
        
        For scalar operators, transpose equals forward.
        
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
        
        For scalar operators, conjugate transpose equals conjugate.
        
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
        
        For scalar operators, transpose equals self.
        
        returns:
            result: Scalar
                The transpose linear system.
        """
        return Scalar(self.scalar)

    def conjugate_LinearSystem(self):
        """
        This method returns the conjugate linear system.
        
        returns:
            result: Scalar
                The conjugate linear system.
        """
        return Scalar(torch.conj(self.scalar))

    def conjugate_transpose_LinearSystem(self):
        """
        This method returns the conjugate transpose linear system.
        
        returns:
            result: Scalar
                The conjugate transpose linear system.
        """
        return Scalar(torch.conj(self.scalar))

# Removed inline test definitions and __main__ execution block. 