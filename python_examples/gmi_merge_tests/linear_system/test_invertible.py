"""
Tests for the InvertibleLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.invertible import InvertibleLinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestInvertibleLinearSystem:
    """Test cases for the InvertibleLinearSystem class."""
    
    def test_abstract_class_instantiation(self):
        """Test that InvertibleLinearSystem cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            InvertibleLinearSystem()
    
    def test_concrete_subclass_implementation(self, sample_tensor_2d):
        """Test that a concrete subclass works correctly."""
        class SimpleInvertibleOperator(InvertibleLinearSystem):
            def __init__(self, factor=2.0):
                super().__init__()
                self.factor = factor

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.factor * x
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y / self.factor
        
        op = SimpleInvertibleOperator(factor=3.0)
        x = sample_tensor_2d
        
        # Test forward
        y = op.forward(x)
        expected_forward = 3.0 * x
        assert torch.allclose(y, expected_forward), f"Forward mismatch: {y} vs {expected_forward}"
        
        # Test inverse
        x_recovered = op.inverse(y)
        assert torch.allclose(x_recovered, x), f"Inverse mismatch: {x_recovered} vs {x}"
        
        # Test that forward and inverse are truly inverses
        x_double_recovered = op.inverse(op.forward(x))
        assert torch.allclose(x_double_recovered, x), "Double inverse should recover original"
    
    def test_square_property_inheritance(self, sample_tensor_3d):
        """Test that InvertibleLinearSystem inherits square property correctly."""
        class SquareInvertibleOperator(InvertibleLinearSystem):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x  # Identity operator
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y  # Identity operator
        
        op = SquareInvertibleOperator()
        
        # Test square property: input_shape = output_shape
        output_shape = (2, 3, 4)
        input_shape = op.input_shape_given_output_shape(output_shape)
        assert input_shape == output_shape, f"Expected {output_shape}, got {input_shape}"
        
        # Test that it's a square operator
        from ct_laboratory.linear_system.square import SquareLinearSystem
        assert isinstance(op, SquareLinearSystem)
    
    def test_inverse_linear_operator_method(self, sample_tensor_2d):
        """Test the inverse_LinearSystem method."""
        class ScaleInvertibleOperator(InvertibleLinearSystem):
            def __init__(self, scale=2.0):
                super().__init__()
                self.scale = scale

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.scale * x
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y / self.scale
        
        op = ScaleInvertibleOperator(scale=2.0)
        x = sample_tensor_2d
        
        # Test that inverse_LinearSystem returns an InverseLinearSystem
        inverse_op = op.inverse_LinearSystem()
        from ct_laboratory.linear_system.inverse import InverseLinearSystem
        assert isinstance(inverse_op, InverseLinearSystem)
        
        # Test that the inverse operator works correctly
        y = op.forward(x)
        x_recovered = inverse_op.forward(y)
        assert torch.allclose(x_recovered, x), "InverseLinearSystem should recover original input"
    
    def test_invertible_property_verification(self, sample_vector_2):
        """Test that invertible operators satisfy the invertible property."""
        # Create a simple invertible operator (scalar)
        from ct_laboratory.linear_system.scalar import Scalar
        op = Scalar(scalar=2.0)
        
        # Use the sample vector directly, don't reshape
        x = sample_vector_2
        
        # Apply forward then inverse
        y = op.forward(x)
        x_recovered = op.inverse(y)
        
        # Should recover original input
        assert torch.allclose(x_recovered, x, atol=1e-6), "Invertible operator should recover original input"
        
        # Apply inverse then forward
        y_inverse = op.inverse(x)
        x_recovered_2 = op.forward(y_inverse)
        
        # Should also recover original input
        assert torch.allclose(x_recovered_2, x, atol=1e-6), "Invertible operator should recover original input in both directions"
    
    def test_inheritance_from_square_operator(self, sample_tensor_2d):
        """Test that InvertibleLinearSystem inherits correctly from SquareLinearSystem."""
        class IdentityInvertibleOperator(InvertibleLinearSystem):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
            
            def inverse(self, y: torch.Tensor) -> torch.Tensor:
                return y
        
        op = IdentityInvertibleOperator()
        x = sample_tensor_2d
        
        # Test that it's a SquareLinearSystem
        from ct_laboratory.linear_system.square import SquareLinearSystem
        assert isinstance(op, SquareLinearSystem)
        
        # Test that it's a LinearSystem
        from ct_laboratory.linear_system.base import LinearSystem
        assert isinstance(op, LinearSystem)
        
        # Test basic operations
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
        
        x_recovered = op.inverse(y)
        assert torch.allclose(x_recovered, x), "Identity inverse should preserve input"
    
    def test_config_instantiation(self):
        """Test that InvertibleLinearSystem cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_system/invertible.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg) 