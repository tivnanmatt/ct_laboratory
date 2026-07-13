"""
Tests for the base LinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.base import LinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestLinearSystem:
    """Test cases for the base LinearSystem class."""
    
    def test_abstract_class_instantiation(self):
        """Test that LinearSystem cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LinearSystem()
    
    def test_concrete_subclass_implementation(self, sample_complex_tensor):
        """Test that a concrete subclass works correctly."""
        # Define a concrete subclass with no symmetries
        class DenseMatrixOperator(LinearSystem):
            def __init__(self):
                super().__init__()
                # Define a 2x2 complex matrix with no symmetries
                # [1+i  2+3i]
                # [3+2i  4+i]
                self.matrix = torch.tensor([[1.0+1j, 2.0+3j], [3.0+2j, 4.0+1j]], dtype=torch.complex64)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
            
            def conjugate(self, x: torch.Tensor) -> torch.Tensor:
                return torch.conj(torch.matmul(self.matrix, torch.conj(x)))
            
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.conj(torch.matmul(self.matrix.transpose(-2, -1), torch.conj(y)))
        
        op = DenseMatrixOperator()
        x = sample_complex_tensor[:2]  # Use first 2 elements for 2x2 matrix
        
        # Test forward operation
        y_forward = op.forward(x)
        y_expected = torch.matmul(op.matrix, x)
        assert torch.allclose(y_forward, y_expected, atol=1e-6), f"Forward mismatch: {y_forward} vs {y_expected}"
        
        # Test transpose operation
        y_transpose = op.transpose(x)
        y_transpose_expected = torch.matmul(op.matrix.transpose(-2, -1), x)
        assert torch.allclose(y_transpose, y_transpose_expected, atol=1e-6), f"Transpose mismatch: {y_transpose} vs {y_transpose_expected}"
        
        # Test conjugate operation
        y_conjugate = op.conjugate(x)
        y_conjugate_expected = torch.conj(torch.matmul(op.matrix, torch.conj(x)))
        assert torch.allclose(y_conjugate, y_conjugate_expected, atol=1e-6), f"Conjugate mismatch: {y_conjugate} vs {y_conjugate_expected}"
        
        # Test conjugate_transpose operation
        y_conj_transpose = op.conjugate_transpose(x)
        y_conj_transpose_expected = torch.conj(torch.matmul(op.matrix.transpose(-2, -1), torch.conj(x)))
        assert torch.allclose(y_conj_transpose, y_conj_transpose_expected, atol=1e-6), f"Conjugate transpose mismatch: {y_conj_transpose} vs {y_conj_transpose_expected}"
        
        # Test that this matrix has NO symmetries
        assert not torch.allclose(y_forward, y_transpose, atol=1e-6), "Should not be symmetric"
        assert not torch.allclose(y_forward, y_conjugate, atol=1e-6), "Should not be real"
        assert not torch.allclose(y_forward, y_conj_transpose, atol=1e-6), "Should not be hermitian"
    
    # def test_pseudoinverse_weighted_average(self, sample_tensor_4d):
    #     """Test the weighted average pseudoinverse method."""
    #     class SimpleOperator(LinearSystem):
    #         def __init__(self):
    #             super().__init__()
    #         
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             return 2.0 * x
    #         
    #         def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
    #             return 2.0 * y
    #     
    #     op = SimpleOperator()
    #     y = sample_tensor_4d
    #     
    #     # Test pseudoinverse
    #     x_pinv = op.pseudoinverse(y, method='weighted_average')
    #     
    #     # For this simple operator, pseudoinverse should be approximately y/4
    #     expected = y / 4.0
    #     assert torch.allclose(x_pinv, expected, atol=1e-6), f"Pseudoinverse mismatch: {x_pinv} vs {expected}"
    
    # def test_pseudoinverse_conjugate_gradient(self, sample_tensor_4d):
    #     """Test the conjugate gradient pseudoinverse method."""
    #     class SimpleOperator(LinearSystem):
    #         def __init__(self):
    #             super().__init__()
    #         
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             return 2.0 * x
    #         
    #         def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
    #             return 2.0 * y
    #     
    #     op = SimpleOperator()
    #     y = sample_tensor_4d
    #     
    #     # Test pseudoinverse with conjugate gradient
    #     x_pinv = op.pseudoinverse(y, method='conjugate_gradient', max_iter=10, tol=1e-6)
    #     
    #     # For this simple operator, pseudoinverse should be approximately y/4
    #     expected = y / 4.0
    #     assert torch.allclose(x_pinv, expected, atol=1e-3), f"Pseudoinverse mismatch: {x_pinv} vs {expected}"
    
    # def test_pseudoinverse_invalid_method(self, sample_tensor_4d):
    #     """Test that pseudoinverse raises error for invalid method."""
    #     class SimpleOperator(LinearSystem):
    #         def __init__(self):
    #             super().__init__()
    #         
    #         def forward(self, x: torch.Tensor) -> torch.Tensor:
    #             return x
    #         
    #         def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
    #             return y
    #     
    #     op = SimpleOperator()
    #     y = sample_tensor_4d
    #     
    #     with pytest.raises(AssertionError, match="The method should be either 'weighted_average' or 'conjugate_gradient'"):
    #         op.pseudoinverse(y, method='invalid_method')
    
    def test_multiplication_operator(self, sample_tensor_4d):
        """Test the multiplication operator overload."""
        class SimpleOperator(LinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return 2.0 * x
        
        op = SimpleOperator()
        x = sample_tensor_4d
        
        # Test multiplication operator
        y1 = op * x
        y2 = op.forward(x)
        
        assert torch.allclose(y1, y2), f"Multiplication operator mismatch: {y1} vs {y2}"
    
    def test_matmul_operator(self):
        """Test the matrix multiplication operator overload."""
        class SimpleOperator(LinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
        
        op1 = SimpleOperator()
        op2 = SimpleOperator()
        
        # Test matrix multiplication operator
        composite = op1 @ op2
        
        from ct_laboratory.linear_system.composite import CompositeLinearSystem
        assert isinstance(composite, CompositeLinearSystem)
    
    def test_unimplemented_methods(self):
        """Test that unimplemented matrix operations raise NotImplementedError."""
        class SimpleOperator(LinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
        
        op = SimpleOperator()
        
        with pytest.raises(NotImplementedError):
            op.mat_add(None)
        
        with pytest.raises(NotImplementedError):
            op.mat_sub(None)
        
        with pytest.raises(NotImplementedError):
            op.mat_mul(None)
    
    def test_config_instantiation(self):
        """Test that LinearSystem cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_system/base.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg) 