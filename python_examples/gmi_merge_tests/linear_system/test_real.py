"""
Tests for the RealLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.real import RealLinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestRealLinearSystem:
    """Test cases for the RealLinearSystem class."""
    
    def test_abstract_class_instantiation(self):
        """Test that RealLinearSystem cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RealLinearSystem()
    
    def test_concrete_subclass_implementation(self, sample_vector_2):
        """Test that a concrete subclass works correctly."""
        class DoubleOperator(RealLinearSystem):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2.0 * x
        
        double_op = DoubleOperator()
        x = sample_vector_2
        
        # Test forward operation
        y = double_op.forward(x)
        expected = 2.0 * x
        assert torch.allclose(y, expected), f"Expected {expected}, got {y}"
        
        # Test real operator property: conjugate = forward
        y_conj = double_op.conjugate(x)
        assert torch.allclose(y_conj, y), f"Expected conjugate to equal forward for real operator"
    
    def test_conjugate_equals_forward(self, sample_vector_2):
        """Test that conjugate operation equals forward operation for real operators."""
        class ScaleOperator(RealLinearSystem):
            def __init__(self, scale=3.0):
                super().__init__()
                self.scale = scale

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.scale * x
        
        op = ScaleOperator(scale=3.0)
        x = sample_vector_2
        
        # Test that conjugate equals forward
        y_forward = op.forward(x)
        y_conjugate = op.conjugate(x)
        
        assert torch.allclose(y_forward, y_conjugate), f"Conjugate should equal forward for real operator: {y_forward} vs {y_conjugate}"
    
    def test_real_operator_properties(self, sample_vector_2):
        """Test that real operators maintain real-valued properties."""
        class IdentityOperator(RealLinearSystem):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
        
        op = IdentityOperator()
        x = sample_vector_2
        
        # Test that output is real-valued
        y = op.forward(x)
        assert torch.isreal(y).all(), "Output should be real-valued for real operator"
        
        # Test conjugate operation
        y_conj = op.conjugate(x)
        assert torch.isreal(y_conj).all(), "Conjugate output should be real-valued for real operator"
        
        # Test that conjugate equals forward
        assert torch.allclose(y, y_conj), "Conjugate should equal forward for real operator"
    
    def test_complex_input_handling(self, sample_complex_tensor):
        """Test that real operators can handle complex inputs correctly."""
        class RealScaleOperator(RealLinearSystem):
            def __init__(self, scale=2.0):
                super().__init__()
                self.scale = scale

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # For real operators, we apply the operation to the real part
                return self.scale * torch.real(x)
        
        op = RealScaleOperator(scale=2.0)
        x = sample_complex_tensor[:2]  # Use first 2 elements for 2x2
        
        # Test forward operation
        y = op.forward(x)
        expected = 2.0 * torch.real(x)
        assert torch.allclose(y, expected), f"Expected {expected}, got {y}"
        
        # Test conjugate operation
        y_conj = op.conjugate(x)
        assert torch.allclose(y_conj, y), "Conjugate should equal forward for real operator"
    
    def test_transpose_operation(self, sample_vector_2, sample_matrix_2x2):
        """Test transpose operation for real operators."""
        class MatrixOperator(RealLinearSystem):
            def __init__(self):
                super().__init__()
                self.matrix = sample_matrix_2x2

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        
        op = MatrixOperator()
        x = sample_vector_2
        
        # Test forward operation
        y_forward = op.forward(x)
        
        # Test transpose operation
        y_transpose = op.transpose(x)
        
        # For real matrices, transpose should be different from forward (unless symmetric)
        assert not torch.allclose(y_forward, y_transpose), "Transpose should be different from forward for non-symmetric matrix"
        
        # Test conjugate operation
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, y_forward), "Conjugate should equal forward for real operator"
    
    def test_config_instantiation(self):
        """Test that RealLinearSystem cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_system/real.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg) 