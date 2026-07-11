import torch
from .composite import CompositeLinearSystem
from .unitary import UnitaryLinearSystem
from .diagonal import DiagonalScalar
from .invertible import InvertibleLinearSystem
from .conjugate_transpose import ConjugateTransposeLinearSystem


class SingularValueDecompositionLinearSystem(CompositeLinearSystem):
    """
    This class represents a linear system given by its singular value decomposition (SVD).

    The operator is constructed as: A = U * S * V^H
    where U and V are unitary matrices and S is a diagonal matrix of singular values.

    parameters:
        singular_value_matrix: DiagonalScalar object
            The diagonal matrix of singular values.
        left_singular_vector_matrix: InvertibleLinearSystem object
            The left singular vector matrix (U).
        right_singular_vector_matrix: InvertibleLinearSystem object
            The right singular vector matrix (V).
    """
    def __init__(self, singular_value_matrix: DiagonalScalar, left_singular_vector_matrix: UnitaryLinearSystem, right_singular_vector_matrix: UnitaryLinearSystem):
        assert isinstance(singular_value_matrix, DiagonalScalar), "The singular values should be a DiagonalScalar object."
        assert isinstance(left_singular_vector_matrix, UnitaryLinearSystem), "The left singular vectors should be a UnitaryLinearSystem object."
        assert isinstance(right_singular_vector_matrix, UnitaryLinearSystem), "The right singular vectors should be a UnitaryLinearSystem object."
        
        # Construct the composite operator: U * Σ * V^H
        v_conj_transpose = ConjugateTransposeLinearSystem(right_singular_vector_matrix)
        operators = [left_singular_vector_matrix, singular_value_matrix, v_conj_transpose]
        
        # Initialize CompositeLinearSystem with the operators
        CompositeLinearSystem.__init__(self, operators)
        
        self.singular_value_matrix = singular_value_matrix
        self.left_singular_vector_matrix = left_singular_vector_matrix
        self.right_singular_vector_matrix = right_singular_vector_matrix
        
        # Add aliases for backward compatibility with tests
        self.left_singular_vectors = left_singular_vector_matrix
        self.right_singular_vectors = right_singular_vector_matrix

    @property
    def singular_values(self):
        """Return the singular value matrix."""
        return self.singular_value_matrix

    @property
    def is_invertible(self) -> bool:
        """
        Check if this singular value decomposition operator is invertible.
        
        returns:
            bool: True if all singular values are non-zero, False otherwise.
        """
        return self.singular_value_matrix.is_invertible

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.left_singular_vector_matrix.forward(self.singular_value_matrix.forward(self.right_singular_vector_matrix.inverse(x)))

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        # For A = U * Σ * V^H, the conjugate transpose is A^H = V * Σ^H * U^H
        # For real singular values, Σ^H = Σ, so A^H = V * Σ * U^H
        return self.right_singular_vector_matrix.forward(self.singular_value_matrix.forward(self.left_singular_vector_matrix.inverse(y)))

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        # For SVD, transpose is V * S^T * U^T
        return self.right_singular_vector_matrix.forward(self.singular_value_matrix.transpose(self.left_singular_vector_matrix.transpose(y)))

    def mat_add(self, M):
        """
        Matrix addition with another linear system.
        
        parameters:
            M: LinearSystem
                The other linear system to add.
        returns:
            result: LinearSystem
                The sum of the two linear systems.
        """
        # For SVD, we can only add with compatible operators
        # For now, we'll use the composite approach
        from .composite import CompositeLinearSystem
        from .identity import Identity
        
        if isinstance(M, SingularValueDecompositionLinearSystem):
            # If both are SVDs, we can add their singular value matrices
            # but this requires the same singular vector matrices
            if (self.left_singular_vector_matrix == M.left_singular_vector_matrix and 
                self.right_singular_vector_matrix == M.right_singular_vector_matrix):
                new_singular_values = self.singular_value_matrix.mat_add(M.singular_value_matrix)
                return SingularValueDecompositionLinearSystem(
                    new_singular_values, 
                    self.left_singular_vector_matrix, 
                    self.right_singular_vector_matrix
                )
        
        # For other cases, use the composite approach
        return CompositeLinearSystem([self, Identity()]).mat_add(M)

    def mat_sub(self, M):
        """
        Matrix subtraction with another linear system.
        
        parameters:
            M: LinearSystem
                The other linear system to subtract.
        returns:
            result: LinearSystem
                The difference of the two linear systems.
        """
        # For SVD, we can only subtract with compatible operators
        # For now, we'll use the composite approach
        from .composite import CompositeLinearSystem
        from .identity import Identity
        
        if isinstance(M, SingularValueDecompositionLinearSystem):
            # If both are SVDs, we can subtract their singular value matrices
            # but this requires the same singular vector matrices
            if (self.left_singular_vector_matrix == M.left_singular_vector_matrix and 
                self.right_singular_vector_matrix == M.right_singular_vector_matrix):
                new_singular_values = self.singular_value_matrix.mat_sub(M.singular_value_matrix)
                return SingularValueDecompositionLinearSystem(
                    new_singular_values, 
                    self.left_singular_vector_matrix, 
                    self.right_singular_vector_matrix
                )
        
        # For other cases, use the composite approach
        return CompositeLinearSystem([self, Identity()]).mat_sub(M)

    def mat_mul(self, M):
        """
        Matrix multiplication with another linear system.
        
        parameters:
            M: LinearSystem or torch.Tensor
                The other linear system to multiply, or a tensor.
        returns:
            result: LinearSystem or torch.Tensor
                The product of the two linear systems, or the result of applying to tensor.
        """
        if isinstance(M, torch.Tensor):
            return self.forward(M)
        elif isinstance(M, SingularValueDecompositionLinearSystem):
            # If both are SVDs, we can multiply their singular value matrices
            # but this requires the same singular vector matrices
            if (self.left_singular_vector_matrix == M.left_singular_vector_matrix and 
                self.right_singular_vector_matrix == M.right_singular_vector_matrix):
                new_singular_values = self.singular_value_matrix.mat_mul(M.singular_value_matrix)
                return SingularValueDecompositionLinearSystem(
                    new_singular_values, 
                    self.left_singular_vector_matrix, 
                    self.right_singular_vector_matrix
                )
        
        # For other cases, use the composite approach
        from .composite import CompositeLinearSystem
        return CompositeLinearSystem([self, M])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # For SVD, the inverse is V * S^{-1} * U^H
        inv_singular_value_matrix = DiagonalScalar(1.0 / self.singular_value_matrix.diagonal_vector)
        return self.right_singular_vector_matrix.forward(inv_singular_value_matrix.forward(self.left_singular_vector_matrix.inverse(y))) 