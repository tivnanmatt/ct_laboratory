"""
Tests for the CompositeLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.composite import CompositeLinearSystem
from ct_laboratory.linear_system.scalar import Scalar

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestCompositeLinearSystem:
    """Test cases for the CompositeLinearSystem class."""
    
    def test_instantiation(self):
        """Test that CompositeLinearSystem can be instantiated directly."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = CompositeLinearSystem(matrix_operators=[op1, op2])
        assert op is not None
        assert len(op.matrix_operators) == 2
    
    def test_forward_operation(self, sample_vector_2):
        """Test that composite operator applies operators in sequence."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = CompositeLinearSystem(matrix_operators=[op1, op2])
        
        x = sample_vector_2
        y = op.forward(x)
        expected = 6.0 * x  # 2.0 * 3.0 * x
        assert torch.allclose(y, expected), "Composite operator should apply operators in sequence"
    
    def test_inheritance(self):
        """Test that CompositeLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        # from ct_laboratory.linear_system.real import RealLinearSystem

        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = CompositeLinearSystem(matrix_operators=[op1, op2])
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        # CompositeLinearSystem should NOT inherit from RealLinearSystem
    
    def test_config_instantiation(self):
        """Test that CompositeLinearSystem can be instantiated from config."""
        cfg = compose(config_name="linear_system/composite.yaml")
        op = instantiate(cfg)
        # The config returns a DictConfig with a linear_system attribute
        op = op.linear_system
        assert isinstance(op, CompositeLinearSystem), "Should instantiate CompositeLinearSystem from config"
    
    def test_composite_properties(self, sample_vector_2):
        """Test that composite operator has all expected properties."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op = CompositeLinearSystem(matrix_operators=[op1, op2])
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 6.0 * x)
        
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
        """Test composite operator with empty list of operators."""
        op = CompositeLinearSystem(matrix_operators=[])
        x = sample_vector_2
        
        # Should act as identity operator
        y = op.forward(x)
        assert torch.allclose(y, x)
    
    def test_single_operator(self, sample_vector_2):
        """Test composite operator with single operator."""
        op1 = Scalar(scalar=2.0)
        op = CompositeLinearSystem(matrix_operators=[op1])
        
        x = sample_vector_2
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
    
    def test_multiple_operators(self, sample_vector_2):
        """Test composite operator with multiple operators."""
        op1 = Scalar(scalar=2.0)
        op2 = Scalar(scalar=3.0)
        op3 = Scalar(scalar=4.0)
        op = CompositeLinearSystem(matrix_operators=[op1, op2, op3])
        
        x = sample_vector_2
        y = op.forward(x)
        expected = 24.0 * x  # 2.0 * 3.0 * 4.0 * x
        assert torch.allclose(y, expected) 