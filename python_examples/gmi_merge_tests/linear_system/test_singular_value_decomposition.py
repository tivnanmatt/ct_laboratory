"""
Tests for the SingularValueDecomposedLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system import SingularValueDecompositionLinearSystem
from ct_laboratory.linear_system import DiagonalScalar, Identity
from ct_laboratory.linear_system.scalar import Scalar
from ct_laboratory.linear_system.conjugate_transpose import ConjugateTransposeLinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestSingularValueDecomposedLinearSystem:
    """Test cases for the SingularValueDecomposedLinearSystem class."""
    
    def test_instantiation(self):
        """Test that SingularValueDecomposedLinearSystem can be instantiated."""
        # Create simple singular value and vector matrices
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        assert op is not None
        assert op.singular_values == singular_value_matrix
        assert op.left_singular_vectors == left_singular_vectors
        assert op.right_singular_vectors == right_singular_vectors
    
    def test_forward_operation(self, sample_vector_2):
        """Test that singular value decomposed operator applies correctly."""
        # For identity singular vectors, the operation should just be diagonal
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        
        x = sample_vector_2
        y = op.forward(x)
        expected = singular_values * x
        assert torch.allclose(y, expected), "Singular value decomposed operator should apply correctly"
    
    def test_inheritance(self):
        """Test that SingularValueDecomposedLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.composite import CompositeLinearSystem
        
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, CompositeLinearSystem), "Should inherit from CompositeLinearSystem"
    
    def test_composite_structure(self, sample_vector_2):
        """Test that the composite structure works correctly."""
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        
        # Should have 3 operators: U, Σ, V^H
        assert len(op.matrix_operators) == 3
        assert op.matrix_operators[0] == left_singular_vectors
        assert op.matrix_operators[1] == singular_value_matrix
        # Check that the third operator is the conjugate transpose of the right singular vectors
        assert isinstance(op.matrix_operators[2], ConjugateTransposeLinearSystem)
    
    def test_invalid_singular_value_matrix_type(self):
        """Test that invalid singular value matrix type raises error."""
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        # Try with a non-diagonal operator
        from ct_laboratory.linear_system.base import LinearSystem
        
        class NonDiagonalOperator(LinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
        
        non_diagonal_op = NonDiagonalOperator()
        
        with pytest.raises(AssertionError, match="The singular values should be a DiagonalScalar object"):
            SingularValueDecompositionLinearSystem(
                singular_value_matrix=non_diagonal_op,
                left_singular_vector_matrix=left_singular_vectors,
                right_singular_vector_matrix=right_singular_vectors
            )
    
    def test_invalid_left_singular_vector_matrix_type(self):
        """Test that invalid left singular vector matrix type raises error."""
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        right_singular_vectors = Identity()
        
        # Try with diagonal operator instead of unitary
        diagonal_op = DiagonalScalar(diagonal_vector=torch.tensor([1.0, 1.0]))
        
        with pytest.raises(AssertionError, match="The left singular vectors should be a UnitaryLinearSystem object"):
            SingularValueDecompositionLinearSystem(
                singular_value_matrix=singular_value_matrix,
                left_singular_vector_matrix=diagonal_op,
                right_singular_vector_matrix=right_singular_vectors
            )
    
    def test_invalid_right_singular_vector_matrix_type(self):
        """Test that invalid right singular vector matrix type raises error."""
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        
        # Try with diagonal operator instead of unitary
        diagonal_op = DiagonalScalar(diagonal_vector=torch.tensor([1.0, 1.0]))
        
        with pytest.raises(AssertionError, match="The right singular vectors should be a UnitaryLinearSystem object"):
            SingularValueDecompositionLinearSystem(
                singular_value_matrix=singular_value_matrix,
                left_singular_vector_matrix=left_singular_vectors,
                right_singular_vector_matrix=diagonal_op
            )
    
    def test_transpose_operation(self, sample_vector_2):
        """Test transpose operation."""
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        
        x = sample_vector_2
        y_transpose = op.transpose(x)
        
        # For identity singular vectors, transpose should equal forward
        y_forward = op.forward(x)
        assert torch.allclose(y_transpose, y_forward)
    
    def test_conjugate_operation(self, sample_vector_2):
        """Test conjugate operation."""
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        
        x = sample_vector_2
        y_conjugate = op.conjugate(x)
        
        # For real operators, conjugate should equal forward
        y_forward = op.forward(x)
        assert torch.allclose(y_conjugate, y_forward)
    
    def test_conjugate_transpose_operation(self, sample_vector_2):
        """Test conjugate transpose operation."""
        singular_values = torch.tensor([2.0, 3.0])
        singular_value_matrix = DiagonalScalar(diagonal_vector=singular_values)
        left_singular_vectors = Identity()
        right_singular_vectors = Identity()
        
        op = SingularValueDecompositionLinearSystem(singular_value_matrix, left_singular_vectors, right_singular_vectors)
        
        x = sample_vector_2
        y_conj_transpose = op.conjugate_transpose(x)
        
        # For identity singular vectors, conjugate transpose should equal forward
        y_forward = op.forward(x)
        assert torch.allclose(y_conj_transpose, y_forward)

def test_is_invertible_with_nonzero_singular_values():
    """Test that is_invertible returns True when all singular values are non-zero."""
    svals = torch.tensor([2.0, 3.0, 4.0, 5.0])
    u = Identity()
    v = Identity()
    op = SingularValueDecompositionLinearSystem(DiagonalScalar(svals), u, v)
    assert op.is_invertible == True

def test_is_invertible_with_zero_singular_values():
    """Test that is_invertible returns False when any singular value is zero."""
    svals = torch.tensor([2.0, 0.0, 4.0, 5.0])  # One zero singular value
    u = Identity()
    v = Identity()
    op = SingularValueDecompositionLinearSystem(DiagonalScalar(svals), u, v)
    assert op.is_invertible == False

def test_is_invertible_with_all_zero_singular_values():
    """Test that is_invertible returns False when all singular values are zero."""
    svals = torch.tensor([0.0, 0.0, 0.0, 0.0])  # All zero singular values
    u = Identity()
    v = Identity()
    op = SingularValueDecompositionLinearSystem(DiagonalScalar(svals), u, v)
    assert op.is_invertible == False 