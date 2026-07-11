"""
Tests for the SymmetricLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.symmetric import SymmetricLinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestSymmetricLinearSystem:
    """Test cases for the SymmetricLinearSystem class."""
    
    def test_abstract_class_instantiation(self):
        """Test that SymmetricLinearSystem cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SymmetricLinearSystem()
    
    def test_concrete_symmetric_operator(self, sample_vector_2, sample_matrix_2x2):
        """Test a symmetric operator: transpose equals forward."""
        class SymmetricOperator(SymmetricLinearSystem):
            def __init__(self):
                super().__init__()
                self.matrix = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
        op = SymmetricOperator()
        x = sample_vector_2
        y_forward = op.forward(x)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_forward, y_transpose, atol=1e-6), "Symmetric operator: transpose should equal forward"
    
    def test_concrete_complex_symmetric_operator(self, sample_vector_2):
        """Test a complex symmetric operator: transpose equals forward."""
        class ComplexSymmetricOperator(SymmetricLinearSystem):
            def __init__(self):
                super().__init__()
                self.matrix = torch.tensor([[1.0+1j, 2.0+3j], [2.0+3j, 4.0+5j]], dtype=torch.complex64)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
        op = ComplexSymmetricOperator()
        x = sample_vector_2.to(torch.complex64)
        y_forward = op.forward(x)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_forward, y_transpose, atol=1e-6), "Symmetric operator: transpose should equal forward"
    
    def test_non_symmetric_matrix(self, sample_vector_2):
        """Test that a non-symmetric matrix does not pass the symmetry check (transpose != forward)."""
        class NonSymmetricOperator(SymmetricLinearSystem):
            def __init__(self):
                super().__init__()
                self.matrix = torch.tensor([[1.0, 2.0], [0.0, 3.0]], dtype=torch.float32)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        op = NonSymmetricOperator()
        x = sample_vector_2
        y_forward = op.forward(x)
        y_transpose = op.transpose(x)
        assert not torch.allclose(y_forward, y_transpose, atol=1e-6), "Non-symmetric: transpose should not equal forward"
    
    def test_inheritance(self, sample_vector_2):
        """Test that SymmetricLinearSystem inherits from LinearSystem and SquareLinearSystem."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.square import SquareLinearSystem
        class SimpleSymmetricOperator(SymmetricLinearSystem):
            def __init__(self):
                super().__init__()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2.0 * x
        op = SimpleSymmetricOperator()
        x = sample_vector_2
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, SquareLinearSystem), "Should inherit from SquareLinearSystem"
        y = op.forward(x)
        expected = 2.0 * x
        assert torch.allclose(y, expected), f"Expected {expected}, got {y}"
        y_transpose = op.transpose(x)
        assert torch.allclose(y, y_transpose, atol=1e-6), "Symmetric operator: transpose should equal forward"
    
    def test_transpose_LinearSystem(self, sample_vector_2):
        """Test that transpose_LinearSystem returns self for symmetric operators."""
        class SimpleSymmetricOperator(SymmetricLinearSystem):
            def __init__(self):
                super().__init__()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2.0 * x
        op = SimpleSymmetricOperator()
        transpose_op = op.transpose_LinearSystem()
        assert transpose_op is op, "Symmetric operator: transpose_LinearSystem should return self"
        
        # Test that the transpose operator behaves the same
        x = sample_vector_2
        y_original = op.forward(x)
        y_transpose = transpose_op.forward(x)
        assert torch.allclose(y_original, y_transpose, atol=1e-6), "Transpose operator should behave the same as original"
    
    def test_config_instantiation(self):
        """Test that SymmetricLinearSystem cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_system/symmetric.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg) 