"""
Tests for the InverseLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.inverse import InverseLinearSystem
from ct_laboratory.linear_system.scalar import Scalar

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestInverseLinearSystem:
    """Test cases for the InverseLinearSystem class."""
    
    def test_instantiation(self):
        """Test that InverseLinearSystem can be instantiated directly."""
        base_op = Scalar(scalar=2.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        assert op is not None
        assert op.base_matrix_operator == base_op
    
    def test_forward_operation(self, sample_vector_2):
        """Test that inverse operator applies inverse of base operator."""
        base_op = Scalar(scalar=2.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        
        x = sample_vector_2
        y = op.forward(x)
        expected = x / 2.0  # Inverse of scalar multiplication
        assert torch.allclose(y, expected), "Inverse operator should apply inverse of base operator"
    
    def test_inheritance(self):
        """Test that InverseLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.invertible import InvertibleLinearSystem
        
        base_op = Scalar(scalar=2.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, InvertibleLinearSystem), "Should inherit from InvertibleLinearSystem"
    
    def test_config_instantiation(self):
        """Test that InverseLinearSystem can be instantiated from config."""
        cfg = compose(config_name="linear_system/inverse.yaml")
        op = instantiate(cfg)
        # The config returns a DictConfig with a linear_system attribute
        op = op.linear_system
        assert isinstance(op, InverseLinearSystem), "Should instantiate InverseLinearSystem from config"
        assert isinstance(op.base_matrix_operator, Scalar), "Should load base operator from config"
        assert op.base_matrix_operator.scalar == 2.0, "Should load base operator parameters from config"
    
    def test_inverse_properties(self, sample_vector_2):
        """Test that inverse operator has all expected properties."""
        base_op = Scalar(scalar=2.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test forward (applies inverse of base)
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, x / 2.0)
        
        # Test inverse (should apply inverse of inverse = original)
        y_inverse = op.inverse(x)
        assert torch.allclose(y_inverse, 2.0 * x)
        
        # Note: conjugate and conjugate_transpose methods are not implemented
        # for InverseLinearSystem as they require additional mathematical
        # properties that may not be available for all base operators
    
    def test_inverse_of_inverse(self, sample_vector_2):
        """Test that inverse of inverse equals original."""
        base_op = Scalar(scalar=2.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Apply inverse operator
        y = op.forward(x)
        assert torch.allclose(y, x / 2.0)
        
        # Apply inverse of inverse (should equal original base operator)
        op_inverse = InverseLinearSystem(base_matrix_operator=op)
        y_inverse_of_inverse = op_inverse.forward(x)
        assert torch.allclose(y_inverse_of_inverse, 2.0 * x)
    
    def test_with_different_base_operator(self, sample_vector_2):
        """Test inverse operator with different base operators."""
        x = sample_vector_2
        
        # Test with different scalar
        base_op = Scalar(scalar=3.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, x / 3.0)
        
        # Test with identity operator
        from ct_laboratory.linear_system.identity import Identity
        base_op = Identity()
        op = InverseLinearSystem(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, x)
    
    def test_forward_inverse_consistency(self, sample_vector_2):
        """Test that forward and inverse operations are consistent."""
        base_op = Scalar(scalar=2.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        
        x = sample_vector_2
        
        # Apply base operator then inverse
        y_base = base_op.forward(x)
        x_recovered = op.forward(y_base)
        assert torch.allclose(x_recovered, x, atol=1e-6)
        
        # Apply inverse then base operator
        y_inverse = op.forward(x)
        x_recovered_2 = base_op.forward(y_inverse)
        assert torch.allclose(x_recovered_2, x, atol=1e-6)
    
    def test_zero_scalar_error(self):
        """Test that inverse operator raises error for non-invertible base operator."""
        base_op = Scalar(scalar=0.0)
        op = InverseLinearSystem(base_matrix_operator=base_op)
        
        x = torch.tensor([1.0, 2.0])
        with pytest.raises(Exception):
            op.forward(x) 