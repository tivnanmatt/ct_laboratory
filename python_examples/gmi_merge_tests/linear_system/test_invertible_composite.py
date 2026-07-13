"""
Tests for the InvertibleCompositeLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.invertible_composite import InvertibleCompositeLinearSystem
from ct_laboratory.linear_system.scalar import Scalar

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestInvertibleCompositeLinearSystem:
    """Test cases for the InvertibleCompositeLinearSystem class."""
    
    def test_instantiation(self):
        """Test that InvertibleCompositeLinearSystem can be instantiated directly."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        assert op is not None
        assert len(op.matrix_operators) == 2
    
    def test_forward_operation(self, sample_vector_2):
        """Test that invertible composite operator applies operators in sequence."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        
        x = sample_vector_2
        y = op.forward(x)
        expected = 6.0 * x  # 2.0 * 3.0 * x
        assert torch.allclose(y, expected), "Invertible composite operator should apply operators in sequence"
    
    def test_inheritance(self):
        """Test that InvertibleCompositeLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.invertible import InvertibleLinearSystem
        
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, InvertibleLinearSystem), "Should inherit from InvertibleLinearSystem"
    
    def test_is_invertible_property(self):
        """Test the is_invertible property for various combinations."""
        # Test with all invertible operators
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        assert op.is_invertible, "Composite with all invertible operators should be invertible"
        
        # Test with non-invertible operator (zero scalar)
        op_zero = Scalar(scalar=0.0)
        op_mixed = InvertibleCompositeLinearSystem(matrix_operators=[op1, op_zero])
        assert not op_mixed.is_invertible, "Composite with non-invertible operator should not be invertible"
        
        # Test with single invertible operator
        op_single = InvertibleCompositeLinearSystem(matrix_operators=[op1])
        assert op_single.is_invertible, "Single invertible operator should be invertible"
        
        # Test with single non-invertible operator
        op_single_zero = InvertibleCompositeLinearSystem(matrix_operators=[op_zero])
        assert not op_single_zero.is_invertible, "Single non-invertible operator should not be invertible"
        
        # Test with multiple non-invertible operators
        op_all_zero = InvertibleCompositeLinearSystem(matrix_operators=[op_zero, op_zero])
        assert not op_all_zero.is_invertible, "All non-invertible operators should not be invertible"
        
        # Test with mixed invertible and non-invertible operators
        op_mixed_order = InvertibleCompositeLinearSystem(matrix_operators=[op_zero, op1, op2])
        assert not op_mixed_order.is_invertible, "Mixed operators with non-invertible should not be invertible"
    
    def test_config_instantiation(self):
        """Test that InvertibleCompositeLinearSystem can be instantiated from config."""
        cfg = compose(config_name="linear_system/invertible_composite.yaml")
        op = instantiate(cfg)
        # The config returns a DictConfig with a linear_system attribute
        op = op.linear_system
        assert isinstance(op, InvertibleCompositeLinearSystem), "Should instantiate InvertibleCompositeLinearSystem from config"
        assert len(op.matrix_operators) == 2, "Should load two operators from config"
    
    def test_invertible_composite_properties(self, sample_vector_2):
        """Test that invertible composite operator has all expected properties."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 6.0 * x)
        
        # Test inverse (should reverse order and invert each)
        y_inverse = op.inverse(x)
        expected_inverse = x / 6.0  # x / (2.0 * 3.0)
        assert torch.allclose(y_inverse, expected_inverse)
        
        # Test transpose (should reverse order and transpose each)
        y_transpose = op.transpose(x)
        # For scalar operators, transpose equals forward, so result should be same
        assert torch.allclose(y_transpose, 6.0 * x)
        
        # Test conjugate (should conjugate each operator)
        y_conjugate = op.conjugate(x)
        # For real scalar operators, conjugate equals forward
        assert torch.allclose(y_conjugate, 6.0 * x)
        
        # Test conjugate_transpose (should reverse order and conjugate_transpose each)
        y_conj_transpose = op.conjugate_transpose(x)
        # For real scalar operators, conjugate_transpose equals forward
        assert torch.allclose(y_conj_transpose, 6.0 * x)
    
    def test_empty_composite(self, sample_vector_2):
        """Test invertible composite operator with empty list of operators."""
        # Note: This should raise an error since InvertibleCompositeLinearSystem requires at least one operator
        with pytest.raises(AssertionError, match="At least one operator should be provided"):
            op = InvertibleCompositeLinearSystem(matrix_operators=[])
    
    def test_single_operator(self, sample_vector_2):
        """Test invertible composite operator with single operator."""
        op1 = Scalar(scalar=2.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1])
        
        x = sample_vector_2
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
        
        y_inverse = op.inverse(x)
        assert torch.allclose(y_inverse, x / 2.0)
    
    def test_multiple_operators(self, sample_vector_2):
        """Test invertible composite operator with multiple operators."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op3 = Scalar(scalar=4.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2, op3])
        
        x = sample_vector_2
        y = op.forward(x)
        expected = 24.0 * x  # 2.0 * 3.0 * 4.0 * x
        assert torch.allclose(y, expected)
        
        y_inverse = op.inverse(x)
        expected_inverse = x / 24.0
        assert torch.allclose(y_inverse, expected_inverse)
    
    def test_forward_inverse_consistency(self, sample_vector_2):
        """Test that forward and inverse operations are consistent."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        
        x = sample_vector_2
        
        # Apply forward then inverse
        y = op.forward(x)
        x_recovered = op.inverse(y)
        assert torch.allclose(x_recovered, x, atol=1e-6)
        
        # Apply inverse then forward
        y_inverse = op.inverse(x)
        x_recovered_2 = op.forward(y_inverse)
        assert torch.allclose(x_recovered_2, x, atol=1e-6)
    
    def test_inverse_LinearSystem(self, sample_vector_2):
        """Test the inverse_LinearSystem method."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op2])
        
        x = sample_vector_2
        
        # Get the inverse operator
        inv_op = op.inverse_LinearSystem()
        assert isinstance(inv_op, InvertibleCompositeLinearSystem), "Should return InvertibleCompositeLinearSystem"
        
        # Test that the inverse operator works correctly
        y = op.forward(x)
        x_recovered = inv_op.forward(y)
        assert torch.allclose(x_recovered, x, atol=1e-6)
        
        # Test that the inverse operator has the correct structure
        assert len(inv_op.matrix_operators) == 2, "Inverse should have same number of operators"
        # The operators should be in reverse order and inverted
        assert torch.allclose(inv_op.matrix_operators[0].scalar, torch.tensor(1.0/3.0)), "First inverse operator should be 1/3"
        assert torch.allclose(inv_op.matrix_operators[1].scalar, torch.tensor(1.0/2.0)), "Second inverse operator should be 1/2"
    
    def test_non_invertible_operator_handling(self, sample_vector_2):
        """Test behavior with non-invertible operators."""
        op1 = Scalar(scalar=2.0)
        op_zero = Scalar(scalar=0.0)
        
        # Create composite with non-invertible operator
        op = InvertibleCompositeLinearSystem(matrix_operators=[op1, op_zero])
        
        # Check that is_invertible returns False
        assert not op.is_invertible, "Composite with non-invertible operator should not be invertible"
        
        x = sample_vector_2
        
        # Forward should still work
        y = op.forward(x)
        expected = torch.zeros_like(x)  # 2.0 * 0.0 * x = 0
        assert torch.allclose(y, expected)
        
        # Inverse should raise an error when called
        with pytest.raises(ValueError, match="The scalar is zero"):
            op.inverse(x)
        
        # inverse_LinearSystem should also raise an error
        with pytest.raises(ValueError, match="The scalar is zero"):
            op.inverse_LinearSystem() 