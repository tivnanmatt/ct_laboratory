import torch
from .composite import CompositeLinearSystem
from .diagonal import DiagonalScalar
from .invertible import InvertibleLinearSystem


class EigenDecompositionLinearSystem(CompositeLinearSystem, InvertibleLinearSystem):
    """
    This class represents a linear system that is given by its eigenvalue decomposition.

    It inherits from CompositeLinearSystem.

    The operator is constructed as: A = Q * Λ * Q^(-1)
    where Q is the eigenvector matrix and Λ is the diagonal eigenvalue matrix.

    parameters:
        eigenvalue_matrix: DiagonalScalar object
            The diagonal matrix of eigenvalues.
        eigenvector_matrix: InvertibleLinearSystem object
            The invertible matrix of eigenvectors.
    """
    def __init__(self, eigenvalue_matrix: DiagonalScalar, eigenvector_matrix: InvertibleLinearSystem):
        assert isinstance(eigenvalue_matrix, DiagonalScalar), "The eigenvalues should be a DiagonalScalar object."
        assert isinstance(eigenvector_matrix, InvertibleLinearSystem), "The eigenvectors should be a InvertibleLinearSystem object."
        
        # Create the composite structure: Q * Λ * Q^(-1)
        operators = [eigenvector_matrix, eigenvalue_matrix, eigenvector_matrix.inverse_LinearSystem()]
        
        # Initialize CompositeLinearSystem with the operators
        CompositeLinearSystem.__init__(self, operators)
        
        # Set attributes
        self.eigenvalue_matrix = eigenvalue_matrix
        self.eigenvector_matrix = eigenvector_matrix

    @property
    def is_invertible(self) -> bool:
        """
        Check if this eigen decomposition operator is invertible.
        
        returns:
            bool: True if all eigenvalues are non-zero, False otherwise.
        """
        return self.eigenvalue_matrix.is_invertible

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.eigenvector_matrix.forward(self.eigenvalue_matrix.forward(self.eigenvector_matrix.inverse(x)))


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
        # For eigen decomposition, we can only add with compatible operators
        # For now, we'll use the composite approach
        from .composite import CompositeLinearSystem
        from .identity import Identity
        
        if isinstance(M, EigenDecompositionLinearSystem):
            # If both are eigen decompositions, we can add their eigenvalue matrices
            # but this requires the same eigenvector matrix
            if self.eigenvector_matrix == M.eigenvector_matrix:
                new_eigenvalues = self.eigenvalue_matrix.mat_add(M.eigenvalue_matrix)
                return EigenDecompositionLinearSystem(new_eigenvalues, self.eigenvector_matrix)
        
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
        # For eigen decomposition, we can only subtract with compatible operators
        # For now, we'll use the composite approach
        from .composite import CompositeLinearSystem
        from .identity import Identity
        
        if isinstance(M, EigenDecompositionLinearSystem):
            # If both are eigen decompositions, we can subtract their eigenvalue matrices
            # but this requires the same eigenvector matrix
            if self.eigenvector_matrix == M.eigenvector_matrix:
                new_eigenvalues = self.eigenvalue_matrix.mat_sub(M.eigenvalue_matrix)
                return EigenDecompositionLinearSystem(new_eigenvalues, self.eigenvector_matrix)
        
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
        elif isinstance(M, EigenDecompositionLinearSystem):
            # If both are eigen decompositions, we can multiply their eigenvalue matrices
            # but this requires the same eigenvector matrix
            if self.eigenvector_matrix == M.eigenvector_matrix:
                new_eigenvalues = self.eigenvalue_matrix.mat_mul(M.eigenvalue_matrix)
                return EigenDecompositionLinearSystem(new_eigenvalues, self.eigenvector_matrix)
        
        # For other cases, use the composite approach
        from .composite import CompositeLinearSystem
        return CompositeLinearSystem([self, M])

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # For a diagonalizable operator, the inverse is Q * Λ^{-1} * Q^{-1}
        inv_eigenvalue_matrix = DiagonalScalar(1.0 / self.eigenvalue_matrix.diagonal_vector)
        return self.eigenvector_matrix.forward(inv_eigenvalue_matrix.forward(self.eigenvector_matrix.inverse(y)))

    def transpose_LinearSystem(self):
        """
        This method returns the transpose linear system.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the transpose is Q^(-T) * Λ * Q^T.
        Since Λ is diagonal (and thus symmetric), this simplifies to Q^(-T) * Λ * Q^T.
        
        returns:
            result: EigenDecompositionLinearSystem
                The transpose linear system.
        """
        # For eigen decomposition, transpose is Q^(-T) * Λ * Q^T
        # Since Λ is diagonal and symmetric, we can use the same eigenvalue matrix
        # but with transposed eigenvector matrices
        Q_T = self.eigenvector_matrix.transpose_LinearSystem()
        Q_inv_T = self.eigenvector_matrix.inverse_LinearSystem().transpose_LinearSystem()
        
        # Create new EigenDecompositionLinearSystem with transposed structure
        # Note: This creates Q^(-T) * Λ * Q^T which is equivalent to the transpose
        return EigenDecompositionLinearSystem(self.eigenvalue_matrix, Q_T)

    def inverse_LinearSystem(self):
        """
        This method returns the inverse linear system.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the inverse is Q * Λ^(-1) * Q^(-1).
        
        returns:
            result: EigenDecompositionLinearSystem
                The inverse linear system.
        raises:
            ValueError: If any eigenvalue is zero.
        """
        if not self.is_invertible:
            raise ValueError("The operator is not invertible (has zero eigenvalues).")
        
        # Create inverse eigenvalue matrix
        inv_eigenvalue_matrix = DiagonalScalar(1.0 / self.eigenvalue_matrix.diagonal_vector)
        
        # Create new EigenDecompositionLinearSystem with inverse eigenvalues
        return EigenDecompositionLinearSystem(inv_eigenvalue_matrix, self.eigenvector_matrix)

    def conjugate_LinearSystem(self):
        """
        This method returns the conjugate linear system.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the conjugate is conj(Q) * conj(Λ) * conj(Q)^(-1).
        
        returns:
            result: EigenDecompositionLinearSystem
                The conjugate linear system.
        """
        # Create conjugate eigenvalue matrix
        conj_eigenvalue_matrix = DiagonalScalar(torch.conj(self.eigenvalue_matrix.diagonal_vector))
        
        # Create conjugate eigenvector matrix
        Q_conj = self.eigenvector_matrix.conjugate_LinearSystem()
        
        # Create new EigenDecompositionLinearSystem with conjugate structure
        return EigenDecompositionLinearSystem(conj_eigenvalue_matrix, Q_conj)

    def conjugate_transpose_LinearSystem(self):
        """
        This method returns the conjugate transpose linear system.
        
        For eigen decomposition A = Q * Λ * Q^(-1), the conjugate transpose is Q^(-H) * conj(Λ) * Q^H.
        
        returns:
            result: EigenDecompositionLinearSystem
                The conjugate transpose linear system.
        """
        # For eigen decomposition, conjugate transpose is Q^(-H) * conj(Λ) * Q^H
        # Since Λ is diagonal, we can use the conjugate eigenvalue matrix
        # but with conjugate transposed eigenvector matrices
        Q_H = self.eigenvector_matrix.conjugate_transpose_LinearSystem()
        Q_inv_H = self.eigenvector_matrix.inverse_LinearSystem().conjugate_transpose_LinearSystem()
        
        # Create conjugate eigenvalue matrix
        conj_eigenvalue_matrix = DiagonalScalar(torch.conj(self.eigenvalue_matrix.diagonal_vector))
        
        # Create new EigenDecompositionLinearSystem with conjugate transposed structure
        # Note: This creates Q^(-H) * conj(Λ) * Q^H which is equivalent to the conjugate transpose
        return EigenDecompositionLinearSystem(conj_eigenvalue_matrix, Q_H) 