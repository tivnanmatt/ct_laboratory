"""
Tests for the DiagonalScalar class.
"""
import pytest
import torch
from ct_laboratory.linear_system.diagonal import DiagonalScalar

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestDiagonalScalar:
    """Test cases for the DiagonalScalar class."""
    
    def test_instantiation(self):
        """Test that DiagonalScalar can be instantiated directly."""
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        assert op is not None
        assert torch.allclose(op.diagonal_vector, diagonal)
    
    def test_forward_operation(self, sample_vector_2):
        """Test that diagonal operator multiplies input by diagonal elements."""
        diagonal = torch.tensor([2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        
        x = sample_vector_2
        y = op.forward(x)
        expected = diagonal * x
        assert torch.allclose(y, expected), "Diagonal operator should multiply by diagonal elements"
    
    def test_inheritance(self):
        """Test that DiagonalScalar inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.square import SquareLinearSystem
        from ct_laboratory.linear_system.symmetric import SymmetricLinearSystem
        from ct_laboratory.linear_system.invertible import InvertibleLinearSystem

        op = DiagonalScalar(diagonal_vector=torch.tensor([1.0, 2.0]))
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, SquareLinearSystem), "Should inherit from SquareLinearSystem"
        assert isinstance(op, SymmetricLinearSystem), "Should inherit from SymmetricLinearSystem"
        assert isinstance(op, InvertibleLinearSystem), "Should inherit from InvertibleLinearSystem"
    
    def test_config_instantiation(self):
        """Test that DiagonalScalar can be instantiated from config."""
        cfg = compose(config_name="linear_system/diagonal.yaml")
        op = instantiate(cfg)
        # The config returns a DictConfig with a linear_system attribute
        op = op.linear_system
        assert isinstance(op, DiagonalScalar), "Should instantiate DiagonalScalar from config"
        expected_diagonal = torch.tensor([2.0, 3.0, 4.0])
        assert torch.allclose(op.diagonal_vector, expected_diagonal), "Should load diagonal vector from config"
    
    def test_diagonal_properties(self, sample_vector_2):
        """Test that diagonal operator has all expected properties."""
        diagonal = torch.tensor([2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, diagonal * x)
        
        # Test transpose (should equal forward for diagonal)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_transpose, diagonal * x)
        
        # Test conjugate (should equal forward for real diagonal)
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, diagonal * x)
        
        # Test conjugate_transpose (should equal forward for real diagonal)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, diagonal * x)
        
        # Test inverse (should divide by diagonal elements)
        y_inverse = op.inverse(x)
        assert torch.allclose(y_inverse, x / diagonal)
    
    def test_different_diagonals(self, sample_vector_2):
        """Test diagonal operator with different diagonal values."""
        x = sample_vector_2
        
        # Test with positive diagonal
        diagonal = torch.tensor([1.0, 2.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        y = op.forward(x)
        assert torch.allclose(y, diagonal * x)
        
        # Test with negative diagonal
        diagonal = torch.tensor([-1.0, -2.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        y = op.forward(x)
        assert torch.allclose(y, diagonal * x)
        
        # Test with zero diagonal
        diagonal = torch.tensor([0.0, 1.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        y = op.forward(x)
        expected = torch.tensor([0.0, x[1]])
        assert torch.allclose(y, expected)
    
    def test_shape_validation(self):
        """Test that diagonal operator validates input shapes."""
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        
        # Should work with matching size
        x = torch.tensor([1.0, 2.0, 3.0])
        y = op.forward(x)
        assert y.shape == x.shape
        
        # Should raise error with mismatched size
        x = torch.tensor([1.0, 2.0])  # Wrong size
        with pytest.raises(Exception):
            op.forward(x)
    
    def test_is_invertible_property(self):
        """Test the is_invertible property."""
        # Test invertible diagonal
        op = DiagonalScalar(diagonal_vector=torch.tensor([1.0, 2.0, 3.0]))
        assert op.is_invertible, "Non-zero diagonal should be invertible"
        
        # Test non-invertible diagonal (with zero)
        op_zero = DiagonalScalar(diagonal_vector=torch.tensor([1.0, 0.0, 3.0]))
        assert not op_zero.is_invertible, "Diagonal with zero should not be invertible"
        
        # Test all zeros
        op_all_zero = DiagonalScalar(diagonal_vector=torch.tensor([0.0, 0.0, 0.0]))
        assert not op_all_zero.is_invertible, "All-zero diagonal should not be invertible"
        
        # Test single element
        op_single = DiagonalScalar(diagonal_vector=torch.tensor([2.0]))
        assert op_single.is_invertible, "Single non-zero element should be invertible"
        
        op_single_zero = DiagonalScalar(diagonal_vector=torch.tensor([0.0]))
        assert not op_single_zero.is_invertible, "Single zero element should not be invertible"

    def test_transpose_LinearSystem(self):
        """Test the transpose_LinearSystem method."""
        # Test with real diagonal
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        transpose_op = op.transpose_LinearSystem()
        
        assert isinstance(transpose_op, DiagonalScalar), "Should return DiagonalScalar"
        assert torch.allclose(transpose_op.diagonal_vector, diagonal), "Transpose should preserve diagonal values"
        
        # Test with complex diagonal
        diagonal_complex = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
        op_complex = DiagonalScalar(diagonal_vector=diagonal_complex)
        transpose_op_complex = op_complex.transpose_LinearSystem()
        
        assert isinstance(transpose_op_complex, DiagonalScalar), "Should return DiagonalScalar"
        assert torch.allclose(transpose_op_complex.diagonal_vector, diagonal_complex), "Transpose should preserve complex diagonal values"
        
        # Test that transpose operator works correctly
        x = torch.tensor([1.0, 2.0, 3.0])
        y_original = op.forward(x)
        y_transpose = transpose_op.forward(x)
        assert torch.allclose(y_original, y_transpose), "Transpose operator should give same result as original"

    def test_conjugate_LinearSystem(self):
        """Test the conjugate_LinearSystem method."""
        # Test with real diagonal
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        conjugate_op = op.conjugate_LinearSystem()
        
        assert isinstance(conjugate_op, DiagonalScalar), "Should return DiagonalScalar"
        assert torch.allclose(conjugate_op.diagonal_vector, diagonal), "Conjugate of real diagonal should be unchanged"
        
        # Test with complex diagonal
        diagonal_complex = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
        op_complex = DiagonalScalar(diagonal_vector=diagonal_complex)
        conjugate_op_complex = op_complex.conjugate_LinearSystem()
        
        assert isinstance(conjugate_op_complex, DiagonalScalar), "Should return DiagonalScalar"
        expected_conjugate = torch.conj(diagonal_complex)
        assert torch.allclose(conjugate_op_complex.diagonal_vector, expected_conjugate), "Conjugate should conjugate complex diagonal values"
        
        # Test that conjugate operator works correctly
        x = torch.tensor([1.0, 2.0])
        y_original = op_complex.forward(x)
        y_conjugate = conjugate_op_complex.forward(x)
        expected_conjugate_result = torch.conj(y_original)
        assert torch.allclose(y_conjugate, expected_conjugate_result), "Conjugate operator should conjugate the result"

    def test_conjugate_transpose_LinearSystem(self):
        """Test the conjugate_transpose_LinearSystem method."""
        # Test with real diagonal
        diagonal = torch.tensor([1.0, 2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        conj_transpose_op = op.conjugate_transpose_LinearSystem()
        
        assert isinstance(conj_transpose_op, DiagonalScalar), "Should return DiagonalScalar"
        assert torch.allclose(conj_transpose_op.diagonal_vector, diagonal), "Conjugate transpose of real diagonal should be unchanged"
        
        # Test with complex diagonal
        diagonal_complex = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
        op_complex = DiagonalScalar(diagonal_vector=diagonal_complex)
        conj_transpose_op_complex = op_complex.conjugate_transpose_LinearSystem()
        
        assert isinstance(conj_transpose_op_complex, DiagonalScalar), "Should return DiagonalScalar"
        expected_conj_transpose = torch.conj(diagonal_complex)
        assert torch.allclose(conj_transpose_op_complex.diagonal_vector, expected_conj_transpose), "Conjugate transpose should conjugate complex diagonal values"
        
        # Test that conjugate transpose operator works correctly
        x = torch.tensor([1.0, 2.0])
        y_original = op_complex.forward(x)
        y_conj_transpose = conj_transpose_op_complex.forward(x)
        expected_conj_transpose_result = torch.conj(y_original)
        assert torch.allclose(y_conj_transpose, expected_conj_transpose_result), "Conjugate transpose operator should conjugate the result"

    def test_operator_chain_operations(self, sample_vector_2):
        """Test chaining of the new LinearSystem methods."""
        diagonal = torch.tensor([2.0, 3.0])
        op = DiagonalScalar(diagonal_vector=diagonal)
        x = sample_vector_2
        
        # Test transpose -> inverse
        transpose_op = op.transpose_LinearSystem()
        inv_transpose_op = transpose_op.inverse_LinearSystem()
        assert isinstance(inv_transpose_op, DiagonalScalar), "Should return DiagonalScalar"
        expected_inv = 1.0 / diagonal
        assert torch.allclose(inv_transpose_op.diagonal_vector, expected_inv), "Inverse of transpose should be 1/diagonal"
        
        # Test inverse -> transpose
        inv_op = op.inverse_LinearSystem()
        transpose_inv_op = inv_op.transpose_LinearSystem()
        assert isinstance(transpose_inv_op, DiagonalScalar), "Should return DiagonalScalar"
        assert torch.allclose(transpose_inv_op.diagonal_vector, expected_inv), "Transpose of inverse should be 1/diagonal"
        
        # Test conjugate -> inverse
        conjugate_op = op.conjugate_LinearSystem()
        inv_conjugate_op = conjugate_op.inverse_LinearSystem()
        assert isinstance(inv_conjugate_op, DiagonalScalar), "Should return DiagonalScalar"
        assert torch.allclose(inv_conjugate_op.diagonal_vector, expected_inv), "Inverse of conjugate should be 1/diagonal"
        
        # Test that operations commute correctly
        y1 = inv_transpose_op.forward(x)
        y2 = transpose_inv_op.forward(x)
        assert torch.allclose(y1, y2), "Inverse of transpose should equal transpose of inverse" 