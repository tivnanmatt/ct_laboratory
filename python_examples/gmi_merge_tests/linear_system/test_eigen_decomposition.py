"""
Tests for the EigenDecomposedLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system import EigenDecompositionLinearSystem
from ct_laboratory.linear_system import DiagonalScalar, Identity
from ct_laboratory.linear_system.scalar import Scalar

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestEigenDecomposedLinearSystem:
    """Test cases for the EigenDecomposedLinearSystem class."""
    
    def test_instantiation(self):
        """Test that EigenDecompositionLinearSystem can be instantiated."""
        eigvals = torch.ones(4)
        eigvecs = Identity()
        op = EigenDecompositionLinearSystem(DiagonalScalar(eigvals), eigvecs)
        assert isinstance(op, EigenDecompositionLinearSystem)
    
    def test_forward_operation(self, sample_vector_2):
        """Test that eigen decomposed operator applies correctly."""
        # For identity eigenvectors, the operation should just be diagonal
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        x = sample_vector_2
        y = op.forward(x)
        expected = eigenvalues * x
        assert torch.allclose(y, expected), "Eigen decomposed operator should apply correctly"
    
    def test_inheritance(self):
        """Test that EigenDecomposedLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.composite import CompositeLinearSystem
        
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, CompositeLinearSystem), "Should inherit from CompositeLinearSystem"
    
    def test_symmetric_property(self, sample_vector_2):
        """Test that eigen decomposed operator is symmetric (transpose equals forward)."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        x = sample_vector_2
        
        # For symmetric operators, transpose should equal forward
        y_forward = op.forward(x)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_forward, y_transpose), "Symmetric operator should have transpose equal to forward"
    
    def test_composite_structure(self, sample_vector_2):
        """Test that the composite structure works correctly."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        # Should have 3 operators: Q, Λ, Q^(-1)
        assert len(op.matrix_operators) == 3
        assert op.matrix_operators[0] == eigenvector_matrix
        assert op.matrix_operators[1] == eigenvalue_matrix
        # Check that the third operator is the inverse of the eigenvector matrix
        assert isinstance(op.matrix_operators[2], type(eigenvector_matrix.inverse_LinearSystem()))
    
    def test_invalid_eigenvalue_matrix_type(self):
        """Test that invalid eigenvalue matrix type raises error."""
        eigenvector_matrix = Identity()
        
        # Try with a non-diagonal operator
        from ct_laboratory.linear_system.base import LinearSystem
        
        class NonDiagonalOperator(LinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        non_diagonal_op = NonDiagonalOperator()
        
        with pytest.raises(AssertionError, match="The eigenvalues should be a DiagonalScalar object"):
            EigenDecompositionLinearSystem(
                eigenvalue_matrix=non_diagonal_op,
                eigenvector_matrix=eigenvector_matrix
            )
    
    def test_invalid_eigenvector_matrix_type(self):
        """Test that invalid eigenvector matrix type raises error."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        
        # Try with a non-invertible operator (we'll use a tensor directly)
        from ct_laboratory.linear_system.base import LinearSystem
        
        class NonInvertibleOperator(LinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        non_invertible_op = NonInvertibleOperator()
        
        with pytest.raises(AssertionError, match="The eigenvectors should be a InvertibleLinearSystem object"):
            EigenDecompositionLinearSystem(
                eigenvalue_matrix=eigenvalue_matrix,
                eigenvector_matrix=non_invertible_op
            )
    
    def test_conjugate_operation(self, sample_vector_2):
        """Test conjugate operation."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        x = sample_vector_2
        y_conjugate = op.conjugate(x)
        
        # For real operators, conjugate should equal forward
        y_forward = op.forward(x)
        assert torch.allclose(y_conjugate, y_forward)
    
    def test_conjugate_transpose_operation(self, sample_vector_2):
        """Test conjugate transpose operation."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        x = sample_vector_2
        y_conj_transpose = op.conjugate_transpose(x)
        
        # For symmetric operators, conjugate transpose should equal forward
        y_forward = op.forward(x)
        assert torch.allclose(y_conj_transpose, y_forward)

def test_is_invertible_with_nonzero_eigenvalues():
    """Test that is_invertible returns True when all eigenvalues are non-zero."""
    eigvals = torch.tensor([2.0, 3.0, 4.0, 5.0])
    eigvecs = Identity()
    op = EigenDecompositionLinearSystem(DiagonalScalar(eigvals), eigvecs)
    assert op.is_invertible == True

def test_is_invertible_with_zero_eigenvalues():
    """Test that is_invertible returns False when any eigenvalue is zero."""
    eigvals = torch.tensor([2.0, 0.0, 4.0, 5.0])  # One zero eigenvalue
    eigvecs = Identity()
    op = EigenDecompositionLinearSystem(DiagonalScalar(eigvals), eigvecs)
    assert op.is_invertible == False

def test_is_invertible_with_all_zero_eigenvalues():
    """Test that is_invertible returns False when all eigenvalues are zero."""
    eigvals = torch.tensor([0.0, 0.0, 0.0, 0.0])  # All zero eigenvalues
    eigvecs = Identity()
    op = EigenDecompositionLinearSystem(DiagonalScalar(eigvals), eigvecs)
    assert op.is_invertible == False


class TestEigenDecompositionLinearSystemMethods:
    """Test cases for the new LinearSystem methods in EigenDecompositionLinearSystem."""
    
    def test_transpose_LinearSystem(self, sample_vector_2):
        """Test the transpose_LinearSystem method."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        transpose_op = op.transpose_LinearSystem()
        
        assert isinstance(transpose_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that transpose operator works correctly
        x = sample_vector_2
        y_original = op.forward(x)
        y_transpose = transpose_op.forward(x)
        assert torch.allclose(y_original, y_transpose), "Transpose operator should give same result as original for symmetric operator"

    def test_conjugate_LinearSystem(self, sample_vector_2):
        """Test the conjugate_LinearSystem method."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        conjugate_op = op.conjugate_LinearSystem()
        
        assert isinstance(conjugate_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that conjugate operator works correctly
        x = sample_vector_2
        y_original = op.forward(x)
        y_conjugate = conjugate_op.forward(x)
        assert torch.allclose(y_original, y_conjugate), "Conjugate operator should give same result as original for real operator"

    def test_conjugate_transpose_LinearSystem(self, sample_vector_2):
        """Test the conjugate_transpose_LinearSystem method."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        conj_transpose_op = op.conjugate_transpose_LinearSystem()
        
        assert isinstance(conj_transpose_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that conjugate transpose operator works correctly
        x = sample_vector_2
        y_original = op.forward(x)
        y_conj_transpose = conj_transpose_op.forward(x)
        assert torch.allclose(y_original, y_conj_transpose), "Conjugate transpose operator should give same result as original for symmetric operator"

    def test_inverse_LinearSystem(self, sample_vector_2):
        """Test the inverse_LinearSystem method."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        inverse_op = op.inverse_LinearSystem()
        
        assert isinstance(inverse_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that inverse operator works correctly
        x = sample_vector_2
        y_original = op.forward(x)
        y_inverse = inverse_op.forward(y_original)
        assert torch.allclose(x, y_inverse, atol=1e-6), "Inverse operator should recover original input"

    def test_operator_chain_operations(self, sample_vector_2):
        """Test chaining of the new LinearSystem methods."""
        eigenvalues = torch.tensor([2.0, 3.0])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        x = sample_vector_2
        
        # Test transpose -> inverse
        transpose_op = op.transpose_LinearSystem()
        inv_transpose_op = transpose_op.inverse_LinearSystem()
        assert isinstance(inv_transpose_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test inverse -> transpose
        inv_op = op.inverse_LinearSystem()
        transpose_inv_op = inv_op.transpose_LinearSystem()
        assert isinstance(transpose_inv_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test conjugate -> inverse
        conjugate_op = op.conjugate_LinearSystem()
        inv_conjugate_op = conjugate_op.inverse_LinearSystem()
        assert isinstance(inv_conjugate_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that operations commute correctly
        y1 = inv_transpose_op.forward(x)
        y2 = transpose_inv_op.forward(x)
        assert torch.allclose(y1, y2, atol=1e-6), "Inverse of transpose should equal transpose of inverse"

    def test_complex_eigenvalues(self, sample_vector_2):
        """Test with complex eigenvalues."""
        eigenvalues = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
        eigenvalue_matrix = DiagonalScalar(diagonal_vector=eigenvalues)
        eigenvector_matrix = Identity()
        
        op = EigenDecompositionLinearSystem(eigenvalue_matrix, eigenvector_matrix)
        
        # Test conjugate
        conjugate_op = op.conjugate_LinearSystem()
        x = sample_vector_2
        y_original = op.forward(x)
        y_conjugate = conjugate_op.forward(x)
        expected_conjugate = torch.conj(y_original)
        assert torch.allclose(y_conjugate, expected_conjugate), "Conjugate should conjugate complex eigenvalues"
        
        # Test conjugate transpose
        conj_transpose_op = op.conjugate_transpose_LinearSystem()
        y_conj_transpose = conj_transpose_op.forward(x)
        expected_conj_transpose = torch.conj(y_original)
        assert torch.allclose(y_conj_transpose, expected_conj_transpose), "Conjugate transpose should conjugate complex eigenvalues" 