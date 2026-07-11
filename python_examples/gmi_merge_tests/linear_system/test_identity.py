"""
Tests for the Identity class.
"""
import pytest
import torch
from ct_laboratory.linear_system.identity import Identity

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestIdentity:
    """Test cases for the Identity class."""
    
    def test_instantiation(self):
        """Test that Identity can be instantiated directly."""
        op = Identity()
        assert op is not None
    
    def test_forward_operation(self, sample_vector_2, sample_tensor_4d):
        """Test that identity operator preserves input."""
        op = Identity()
        
        # Test with vector
        x = sample_vector_2
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
        
        # Test with tensor
        x = sample_tensor_4d
        y = op.forward(x)
        assert torch.allclose(y, x), "Identity operator should preserve input"
    
    def test_inheritance(self):
        """Test that Identity inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        from ct_laboratory.linear_system.square import SquareLinearSystem
        from ct_laboratory.linear_system.scalar import Scalar
        from ct_laboratory.linear_system.real import RealLinearSystem
        from ct_laboratory.linear_system.hermitian import HermitianLinearSystem
        from ct_laboratory.linear_system.unitary import UnitaryLinearSystem
        from ct_laboratory.linear_system.invertible import InvertibleLinearSystem
        
        op = Identity()
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        assert isinstance(op, SquareLinearSystem), "Should inherit from SquareLinearSystem"
        assert isinstance(op, Scalar), "Should inherit from Scalar"
        assert isinstance(op, RealLinearSystem), "Should inherit from RealLinearSystem"
        assert isinstance(op, HermitianLinearSystem), "Should inherit from HermitianLinearSystem"
        assert isinstance(op, UnitaryLinearSystem), "Should inherit from UnitaryLinearSystem"
        assert isinstance(op, InvertibleLinearSystem), "Should inherit from InvertibleLinearSystem"
    
    def test_is_invertible_property(self):
        """Test the is_invertible property inherited from Scalar."""
        op = Identity()
        assert op.is_invertible, "Identity operator should be invertible (scalar = 1.0)"
    
    def test_config_instantiation(self):
        """Test that Identity can be instantiated from config."""
        cfg = compose(config_name="linear_system/identity.yaml")
        op = instantiate(cfg)
        # The config returns a DictConfig with a linear_system attribute
        op = op.linear_system
        assert isinstance(op, Identity), "Should instantiate Identity from config"
    
    def test_identity_properties(self, sample_vector_2):
        """Test that identity operator has all expected properties."""
        op = Identity()
        x = sample_vector_2
        
        # Test forward
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, x)
        
        # Test transpose (should equal forward for identity)
        y_transpose = op.transpose(x)
        assert torch.allclose(y_transpose, x)
        
        # Test conjugate (should equal forward for real identity)
        y_conjugate = op.conjugate(x)
        assert torch.allclose(y_conjugate, x)
        
        # Test conjugate_transpose (should equal forward for real identity)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, x)
        
        # Test inverse (should equal forward for identity)
        y_inverse = op.inverse(x)
        assert torch.allclose(y_inverse, x)
    
    def test_scalar_property(self):
        """Test that identity operator has scalar property equal to 1.0."""
        op = Identity()
        assert torch.allclose(op.scalar, torch.tensor(1.0)), "Identity operator should have scalar value 1.0"
    
    def test_all_operations_preserve_input(self, sample_vector_2):
        """Test that all operations preserve the input (identity behavior)."""
        op = Identity()
        x = sample_vector_2
        
        # Test all operations return the input unchanged
        operations = [
            op.forward,
            op.transpose,
            op.conjugate,
            op.conjugate_transpose,
            op.inverse
        ]
        
        for operation in operations:
            result = operation(x)
            assert torch.allclose(result, x), f"Operation {operation.__name__} should preserve input"
    
    def test_complex_input_handling(self):
        """Test that identity operator handles complex inputs correctly."""
        op = Identity()
        
        # Test with complex tensor
        x_complex = torch.tensor([1.0 + 2j, 3.0 - 4j])
        y = op.forward(x_complex)
        assert torch.allclose(y, x_complex)
        
        # Test conjugate with complex input
        y_conj = op.conjugate(x_complex)
        assert torch.allclose(y_conj, x_complex)  # Identity preserves complex values
    
    def test_tensor_input_handling(self, sample_tensor_4d):
        """Test that identity operator handles multi-dimensional tensors correctly."""
        op = Identity()
        x = sample_tensor_4d
        
        y = op.forward(x)
        assert torch.allclose(y, x)
        assert y.shape == x.shape
    
    def test_matrix_operations(self, sample_vector_2):
        """Test matrix operations with other operators."""
        from ct_laboratory.linear_system.scalar import Scalar
        from ct_laboratory.linear_system.diagonal import DiagonalScalar
        
        op = Identity()
        x = sample_vector_2
        
        # Test addition with Identity
        other_identity = Identity()
        sum_op = op.mat_add(other_identity)
        assert isinstance(sum_op, Scalar), "Identity + Identity should return Scalar"
        assert torch.allclose(sum_op.scalar, torch.tensor(2.0)), "Identity + Identity should equal 2.0"
        
        # Test addition with Scalar
        scalar_op = Scalar(scalar=3.0)
        sum_op = op.mat_add(scalar_op)
        assert isinstance(sum_op, Scalar), "Identity + Scalar should return Scalar"
        assert torch.allclose(sum_op.scalar, torch.tensor(4.0)), "Identity + 3.0 should equal 4.0"
        
        # Test addition with DiagonalScalar
        diagonal_op = DiagonalScalar(diagonal_vector=torch.tensor([1.0, 2.0]))
        sum_op = op.mat_add(diagonal_op)
        assert isinstance(sum_op, DiagonalScalar), "Identity + Diagonal should return DiagonalScalar"
        expected_diagonal = torch.tensor([2.0, 3.0])  # 1.0 + [1.0, 2.0]
        assert torch.allclose(sum_op.diagonal_vector, expected_diagonal)
        
        # Test subtraction with Identity
        diff_op = op.mat_sub(other_identity)
        assert isinstance(diff_op, Scalar), "Identity - Identity should return Scalar"
        assert torch.allclose(diff_op.scalar, torch.tensor(0.0)), "Identity - Identity should equal 0.0"
        
        # Test subtraction with Scalar
        diff_op = op.mat_sub(scalar_op)
        assert isinstance(diff_op, Scalar), "Identity - Scalar should return Scalar"
        assert torch.allclose(diff_op.scalar, torch.tensor(-2.0)), "Identity - 3.0 should equal -2.0"
        
        # Test subtraction with DiagonalScalar
        diff_op = op.mat_sub(diagonal_op)
        assert isinstance(diff_op, DiagonalScalar), "Identity - Diagonal should return DiagonalScalar"
        expected_diagonal = torch.tensor([0.0, -1.0])  # 1.0 - [1.0, 2.0]
        assert torch.allclose(diff_op.diagonal_vector, expected_diagonal)
        
        # Test multiplication with Identity
        prod_op = op.mat_mul(other_identity)
        assert isinstance(prod_op, Identity), "Identity * Identity should return Identity"
        
        # Test multiplication with Scalar
        prod_op = op.mat_mul(scalar_op)
        assert isinstance(prod_op, Scalar), "Identity * Scalar should return Scalar"
        assert torch.allclose(prod_op.scalar, torch.tensor(3.0)), "Identity * 3.0 should equal 3.0"
        
        # Test multiplication with DiagonalScalar
        prod_op = op.mat_mul(diagonal_op)
        assert isinstance(prod_op, DiagonalScalar), "Identity * Diagonal should return DiagonalScalar"
        assert torch.allclose(prod_op.diagonal_vector, diagonal_op.diagonal_vector), "Identity * Diagonal should preserve diagonal"
        
        # Test multiplication with tensor
        y = op.mat_mul(x)
        assert torch.allclose(y, x), "Identity * tensor should preserve tensor"
    
    def test_matrix_operations_error_handling(self):
        """Test that matrix operations raise appropriate errors for unsupported types."""
        from ct_laboratory.linear_system.conjugate import ConjugateLinearSystem
        
        op = Identity()
        unsupported_op = ConjugateLinearSystem(base_matrix_operator=op)
        
        # Test addition with unsupported type
        with pytest.raises(ValueError, match="Addition with.*not supported"):
            op.mat_add(unsupported_op)
        
        # Test subtraction with unsupported type
        with pytest.raises(ValueError, match="Subtraction with.*not supported"):
            op.mat_sub(unsupported_op)
        
        # Test multiplication with unsupported type
        # Note: mat_mul returns the other operator for unsupported types, so no error
        result = op.mat_mul(unsupported_op)
        assert result == unsupported_op, "Multiplication with unsupported type should return the other operator"
    
    def test_logdet(self):
        """Test log determinant calculation."""
        op = Identity()
        logdet = op.logdet()
        
        expected = torch.tensor(0.0)
        assert torch.allclose(logdet, expected), "Log determinant of identity should be 0.0"
    
    def test_inverse_LinearSystem(self):
        """Test inverse linear operator method."""
        op = Identity()
        inv_op = op.inverse_LinearSystem()
        
        assert isinstance(inv_op, Identity), "Inverse of identity should be identity"
        assert inv_op is not op, "Should return a new instance"
    
    def test_sqrt_LinearSystem(self):
        """Test square root linear operator method."""
        op = Identity()
        sqrt_op = op.sqrt_LinearSystem()
        
        assert isinstance(sqrt_op, Identity), "Square root of identity should be identity"
        assert sqrt_op is not op, "Should return a new instance"
    
    def test_inverse_operations(self, sample_vector_2):
        """Test that inverse operations work correctly."""
        op = Identity()
        x = sample_vector_2
        
        # Test inverse method
        y = op.inverse(x)
        assert torch.allclose(y, x), "Inverse should preserve input"
        
        # Test inverse_LinearSystem method
        inv_op = op.inverse_LinearSystem()
        y_inv = inv_op.forward(x)
        assert torch.allclose(y_inv, x), "Inverse operator should preserve input"
        
        # Test that inverse operator is identity
        assert isinstance(inv_op, Identity), "Inverse operator should be Identity" 