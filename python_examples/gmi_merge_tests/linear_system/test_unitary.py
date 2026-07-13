"""
Tests for the UnitaryLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.unitary import UnitaryLinearSystem

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestUnitaryLinearSystem:
    """Test cases for the UnitaryLinearSystem class."""
    
    def test_abstract_class_instantiation(self):
        """Test that UnitaryLinearSystem cannot be instantiated directly (abstract class)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            UnitaryLinearSystem()
    
    def test_concrete_subclass_implementation(self, sample_tensor_2d):
        """Test that a concrete subclass works correctly."""
        # Test concrete subclass: 2x2 unitary but not real/symmetric/hermitian matrix
        # [cos(θ)  -sin(θ)]
        # [sin(θ)   cos(θ)]  where θ = π/4
        class RotationMatrixOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()
                theta = torch.tensor(torch.pi / 4)
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                # Define the 2x2 rotation matrix
                self.matrix = torch.tensor([[cos_theta.item(), -sin_theta.item()], [sin_theta.item(), cos_theta.item()]], dtype=torch.float32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Apply 2x2 matrix multiplication
                return torch.matmul(self.matrix, x)
            
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                # For real matrices, conjugate_transpose = transpose
                return torch.matmul(self.matrix.transpose(-2, -1), y)
            
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                # For real matrices, transpose = conjugate_transpose
                return self.conjugate_transpose(y)
        
        op = RotationMatrixOperator()
        x = sample_tensor_2d[:2]  # Use first 2 elements for 2x2 matrix
        y = op.forward(x)
        
        # Test that it's unitary: inverse = conjugate_transpose
        y_inverse = op.inverse(y)
        assert torch.allclose(x, y_inverse, atol=1e-6), "Unitary operator: inverse should recover original input"
        
        # Test that it's NOT symmetric (transpose != forward)
        y_transpose = op.transpose(x)
        assert not torch.allclose(y, y_transpose, atol=1e-6), "Should not be symmetric (transpose should not equal forward)"
    
    def test_unitary_property(self, sample_tensor_2d):
        """Test that unitary operators have inverse = conjugate_transpose."""
        class IdentityUnitaryOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()
                # Identity matrix is unitary
                self.matrix = torch.eye(2, dtype=torch.complex64)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.conj().transpose(-2, -1), y)
        op = IdentityUnitaryOperator()
        x = torch.tensor([1.0, 0.0], dtype=torch.complex64)
        y = op.forward(x)
        y_inverse = op.inverse(y)
        assert torch.allclose(x, y_inverse, atol=1e-6), "Identity unitary operator: inverse should recover original input for canonical basis vector"
        # Matrix property check (with a proper unitary matrix)
        class HadamardUnitaryOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()
                # Hadamard matrix is unitary (up to normalization)
                self.matrix = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.conj().transpose(-2, -1), y)
        op2 = HadamardUnitaryOperator()
        identity_check = torch.matmul(op2.matrix.conj().transpose(-2, -1), op2.matrix)
        expected_identity = torch.eye(2, dtype=torch.complex64)
        assert torch.allclose(identity_check, expected_identity, atol=1e-6), "Hadamard unitary matrix should satisfy U^H U = I"
    
    def test_real_unitary_matrix(self, sample_tensor_2d):
        """Test real unitary matrix (orthogonal matrix) properties."""
        class OrthogonalOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()
                # Real orthogonal matrix (real unitary)
                self.matrix = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float32)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                # For real matrices, conjugate_transpose = transpose
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        
        op = OrthogonalOperator()
        x = sample_tensor_2d[:2]
        
        # Test forward operation
        y = op.forward(x)
        
        # Test that it's unitary: inverse = conjugate_transpose
        y_inverse = op.inverse(y)
        assert torch.allclose(x, y_inverse, atol=1e-6), "Orthogonal operator: inverse should recover original input"
        
        # Test that it's orthogonal: A^T A = I
        identity_check = torch.matmul(op.matrix.transpose(-2, -1), op.matrix)
        expected_identity = torch.eye(2, dtype=torch.float32)
        assert torch.allclose(identity_check, expected_identity, atol=1e-6), "Orthogonal matrix should satisfy A^T A = I"
    
    def test_complex_unitary_matrix(self, sample_complex_tensor):
        """Test complex unitary matrix properties."""
        class ComplexUnitaryOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()
                # Truly non-symmetric complex unitary matrix U where U^H U = I and U != U^T
                self.matrix = torch.tensor([[1.0+0j, 1.0j], [1.0+0j, -1.0j]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0))
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix, x)
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.conj().transpose(-2, -1), y)
            def transpose(self, y: torch.Tensor) -> torch.Tensor:
                return torch.matmul(self.matrix.transpose(-2, -1), y)
        op = ComplexUnitaryOperator()
        # Verify unitary property directly on the matrix
        identity_check = torch.matmul(op.matrix.conj().transpose(-2, -1), op.matrix)
        expected_identity = torch.eye(2, dtype=torch.complex64)
        assert torch.allclose(identity_check, expected_identity, atol=1e-6), "Complex unitary matrix should satisfy U^H U = I"

        x = torch.tensor([1.0+0j, 0.5+0.5j], dtype=torch.complex64)
        y = op.forward(x)
        # Test that it's NOT symmetric (transpose produces different result)
        y_transpose = op.transpose(x)
        assert not torch.allclose(y, y_transpose, atol=1e-6), "Complex unitary matrix should not be symmetric in general"
    
    def test_inheritance_from_linear_operator(self, sample_tensor_2d):
        """Test that UnitaryLinearSystem inherits correctly from LinearSystem."""
        class SimpleUnitaryOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x  # Identity is unitary
            
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return y  # Identity is its own conjugate transpose
        
        op = SimpleUnitaryOperator()
        x = sample_tensor_2d
        
        # Test that it's a LinearSystem
        from ct_laboratory.linear_system.base import LinearSystem
        assert isinstance(op, LinearSystem)
        
        # Test basic operations
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
        
        # Test unitary property
        y_inverse = op.inverse(y)
        assert torch.allclose(x, y_inverse, atol=1e-6), "Unitary operator: inverse should recover original input"
    
    def test_config_instantiation(self):
        """Test that UnitaryLinearSystem cannot be instantiated from config (abstract class)."""
        cfg = compose(config_name="linear_system/unitary.yaml")
        with pytest.raises(Exception, match="Can't instantiate abstract class"):
            instantiate(cfg)
    
    def test_inheritance(self):
        """Test that UnitaryLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.square import SquareLinearSystem
        from ct_laboratory.linear_system.invertible import InvertibleLinearSystem

        # Create a concrete subclass for testing inheritance
        class ConcreteUnitaryOperator(UnitaryLinearSystem):
            def __init__(self):
                super().__init__()
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x  # Identity is unitary
            
            def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
                return y  # Identity is its own conjugate transpose
        
        op = ConcreteUnitaryOperator()
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, SquareLinearSystem), "Should inherit from SquareLinearSystem"
        assert isinstance(op, InvertibleLinearSystem), "Should inherit from InvertibleLinearSystem" 