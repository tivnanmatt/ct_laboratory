"""
Tests for the HermitianLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.hermitian import HermitianLinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestHermitianLinearSystem:
    """Test cases for the HermitianLinearSystem class."""
    
    def test_abstract_class_instantiation(self):
        """Test that HermitianLinearSystem cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            HermitianLinearSystem()
    
    def test_concrete_complex_hermitian_operator(self, sample_vector_2):
        """Test a complex hermitian operator: conjugate_transpose equals forward."""
        class ComplexHermitianOperator(HermitianLinearSystem):
            def __init__(self):
                super().__init__()
                # Complex hermitian matrix: A = A^H but A ≠ A^T
                self.matrix = torch.tensor([[1.0, 2.0+1j], [2.0-1j, 3.0]], dtype=torch.complex64)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        
        op = ComplexHermitianOperator()
        x = sample_vector_2.to(torch.complex64)
        
        # Test forward operation
        y_forward = op.forward(x)
        
        # Test hermitian property: conjugate_transpose = forward
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_forward, y_conj_transpose, atol=1e-6), "Hermitian operator: conjugate_transpose should equal forward"
        
        # Test that it's NOT symmetric (transpose ≠ forward for complex hermitian)
        y_transpose = op.transpose(x)
        assert not torch.allclose(y_forward, y_transpose, atol=1e-6), "Complex hermitian: transpose should not equal forward"
    
    def test_concrete_real_hermitian_operator(self, sample_vector_2):
        """Test a real hermitian operator: conjugate_transpose = forward."""
        class RealHermitianOperator(HermitianLinearSystem):
            def __init__(self):
                super().__init__()
                # Real hermitian matrix: A = A^H = A^T
                self.matrix = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        
        op = RealHermitianOperator()
        x = sample_vector_2
        
        # Test forward operation
        y_forward = op.forward(x)
        
        # Test hermitian property: conjugate_transpose = forward
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_forward, y_conj_transpose, atol=1e-6), "Hermitian operator: conjugate_transpose should equal forward"
        
        # Test that it's also symmetric (transpose = forward for real hermitian)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_forward, y_transpose, atol=1e-6), "Real hermitian: transpose should equal forward"
    
    def test_non_hermitian_matrix(self, sample_vector_2):
        """Test that a non-hermitian matrix does not pass the hermitian check (conjugate_transpose != forward)."""
        class NonHermitianOperator(HermitianLinearSystem):
            def __init__(self):
                super().__init__()
                # Non-hermitian matrix: A ≠ A^H
                self.matrix = torch.tensor([[1.0, 2.0+1j], [3.0+1j, 4.0]], dtype=torch.complex64)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.conj().transpose(-2, -1), y)
        
        op = NonHermitianOperator()
        x = sample_vector_2.to(torch.complex64)
        
        # Test that conjugate_transpose does NOT equal forward
        y_forward = op.forward(x)
        y_conj_transpose = op.conjugate_transpose(x)
        assert not torch.allclose(y_forward, y_conj_transpose, atol=1e-6), "Non-hermitian: conjugate_transpose should not equal forward"
    
    def test_inheritance(self, sample_vector_2):
        """Test that HermitianLinearSystem inherits from SquareLinearSystem and LinearSystem."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.square import SquareLinearSystem
        
        class SimpleHermitianOperator(HermitianLinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2.0 * x
        
        op = SimpleHermitianOperator()
        x = sample_vector_2
        
        # Test inheritance
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, SquareLinearSystem), "Should inherit from SquareLinearSystem"
        
        # Test basic operations
        y = op.forward(x)
        expected = 2.0 * x
        assert torch.allclose(y, expected), f"Expected {expected}, got {y}"
        
        # Test hermitian property
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y, y_conj_transpose, atol=1e-6), "Hermitian operator: conjugate_transpose should equal forward"
    
    def test_hermitian_vs_symmetric_behavior(self, sample_vector_2):
        """Test that hermitian and symmetric operators behave differently in general."""
        from ct_laboratory.linear_system.symmetric import SymmetricLinearSystem
        
        class ComplexHermitianOperator(HermitianLinearSystem):
            def __init__(self):
                super().__init__()
                # Complex hermitian matrix
                self.matrix = torch.tensor([[1.0, 2.0+1j], [2.0-1j, 3.0]], dtype=torch.complex64)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        
        class ComplexSymmetricOperator(SymmetricLinearSystem):
            def __init__(self):
                super().__init__()
                # Complex symmetric matrix (not hermitian)
                self.matrix = torch.tensor([[1.0, 2.0+1j], [2.0+1j, 3.0]], dtype=torch.complex64)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
        
        hermitian_op = ComplexHermitianOperator()
        symmetric_op = ComplexSymmetricOperator()
        x = sample_vector_2.to(torch.complex64)
        
        # Test hermitian operator
        y_hermitian = hermitian_op.forward(x)
        y_hermitian_conj_transpose = hermitian_op.conjugate_transpose(x)
        y_hermitian_transpose = hermitian_op.transpose(x)
        
        # Test symmetric operator
        y_symmetric = symmetric_op.forward(x)
        y_symmetric_transpose = symmetric_op.transpose(x)
        
        # Hermitian: conjugate_transpose = forward, but transpose ≠ forward
        assert torch.allclose(y_hermitian, y_hermitian_conj_transpose, atol=1e-6), "Hermitian: conjugate_transpose should equal forward"
        assert not torch.allclose(y_hermitian, y_hermitian_transpose, atol=1e-6), "Hermitian: transpose should not equal forward"
        
        # Symmetric: transpose = forward
        assert torch.allclose(y_symmetric, y_symmetric_transpose, atol=1e-6), "Symmetric: transpose should equal forward"
    
    def test_conjugate_transpose_linear_operator_method(self, sample_vector_2):
        """Test the conjugate_transpose_LinearSystem method."""
        class TestHermitianOperator(HermitianLinearSystem):
            def __init__(self):
                super().__init__()
                self.matrix = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
        
        op = TestHermitianOperator()
        x = sample_vector_2
        
        # Test that conjugate_transpose_LinearSystem returns self
        conj_transpose_op = op.conjugate_transpose_LinearSystem()
        assert conj_transpose_op is op, "conjugate_transpose_LinearSystem should return self"
        
        # Test that the returned operator works correctly
        y = op.forward(x)
        y_conj_transpose = conj_transpose_op.forward(x)
        assert torch.allclose(y, y_conj_transpose, atol=1e-6), "conjugate_transpose_LinearSystem should work correctly"
    
    def test_config_instantiation(self):
        """Test that HermitianLinearSystem cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_system/hermitian.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg) 