"""
Tests for the ConjugateTransposeLinearSystem class.
"""
import pytest
import torch
from ct_laboratory.linear_system.conjugate_transpose import ConjugateTransposeLinearSystem
from ct_laboratory.linear_system.scalar import Scalar

# Global initialization for Hydra
from hydra import compose, initialize
from hydra.utils import instantiate

# Initialize once for all tests in this module
from hydra.core.global_hydra import GlobalHydra
if not GlobalHydra.instance().is_initialized():
    initialize(version_base=None, config_path="../configs")


class TestConjugateTransposeLinearSystem:
    """Test cases for the ConjugateTransposeLinearSystem class."""
    
    def test_instantiation(self):
        """Test that ConjugateTransposeLinearSystem can be instantiated directly."""
        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        assert op is not None
        assert op.base_matrix_operator == base_op
    
    def test_forward_operation(self, sample_vector_2):
        """Test that conjugate transpose operator applies conjugate transpose of base operator."""
        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        
        x = sample_vector_2
        y = op.forward(x)
        # For real scalar operator, conjugate transpose equals forward
        expected = 2.0 * x
        assert torch.allclose(y, expected), "Conjugate transpose operator should apply conjugate transpose of base operator"
    
    def test_inheritance(self):
        """Test that ConjugateTransposeLinearSystem inherits correctly."""
        from ct_laboratory.linear_system.base import LinearSystem
        # from ct_laboratory.linear_system.real import RealLinearSystem

        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        assert isinstance(op, LinearSystem), "Should inherit from LinearSystem"
        # ConjugateTransposeLinearSystem should NOT inherit from RealLinearSystem
    
    def test_config_instantiation(self):
        """Test that ConjugateTransposeLinearSystem can be instantiated from config."""
        cfg = compose(config_name="linear_system/conjugate_transpose.yaml")
        op = instantiate(cfg)
        # The config returns a DictConfig with a linear_system attribute
        op = op.linear_system
        assert isinstance(op, ConjugateTransposeLinearSystem), "Should instantiate ConjugateTransposeLinearSystem from config"
        assert isinstance(op.base_matrix_operator, Scalar), "Should load base operator from config"
        assert op.base_matrix_operator.scalar == 2.0, "Should load base operator parameters from config"
    
    def test_conjugate_transpose_properties(self, sample_vector_2):
        """Test that conjugate transpose operator has all expected properties."""
        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test forward (applies conjugate transpose of base)
        y_forward = op.forward(x)
        assert torch.allclose(y_forward, 2.0 * x)
        
        # Test conjugate_transpose (should apply conjugate_transpose of conjugate_transpose = original)
        y_conj_transpose = op.conjugate_transpose(x)
        assert torch.allclose(y_conj_transpose, 2.0 * x)
        
        # Test transpose (should transpose the conjugate transpose)
        y_transpose = op.transpose(x)
        # For real scalar operator, transpose equals forward
        assert torch.allclose(y_transpose, 2.0 * x)
        
        # Test conjugate (should conjugate the conjugate transpose)
        y_conjugate = op.conjugate(x)
        # For real scalar operator, conjugate equals forward
        assert torch.allclose(y_conjugate, 2.0 * x)
    
    def test_conjugate_transpose_of_conjugate_transpose(self, sample_vector_2):
        """Test that conjugate transpose of conjugate transpose equals original."""
        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Apply conjugate transpose operator
        y = op.forward(x)
        assert torch.allclose(y, 2.0 * x)
        
        # Apply conjugate transpose of conjugate transpose (should equal original base operator)
        op_conj_transpose = ConjugateTransposeLinearSystem(base_matrix_operator=op)
        y_conj_transpose_of_conj_transpose = op_conj_transpose.forward(x)
        assert torch.allclose(y_conj_transpose_of_conj_transpose, 2.0 * x)
    
    def test_with_different_base_operator(self, sample_vector_2):
        """Test conjugate transpose operator with different base operators."""
        x = sample_vector_2
        
        # Test with different scalar
        base_op = Scalar(scalar=3.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, 3.0 * x)
        
        # Test with identity operator
        from ct_laboratory.linear_system.identity import Identity
        base_op = Identity()
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        y = op.forward(x)
        assert torch.allclose(y, x)
    
    def test_with_complex_input(self, sample_vector_2):
        """Test conjugate transpose operator with complex input."""
        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        
        # Create complex input
        x_complex = sample_vector_2 + 1j * sample_vector_2
        y = op.forward(x_complex)
        
        # For real scalar operator, conjugate transpose should preserve complex input
        expected = 2.0 * x_complex
        assert torch.allclose(y, expected)
    
    def test_relationship_with_transpose_and_conjugate(self, sample_vector_2):
        """Test that conjugate transpose equals conjugate of transpose and transpose of conjugate."""
        base_op = Scalar(scalar=2.0)
        op = ConjugateTransposeLinearSystem(base_matrix_operator=base_op)
        x = sample_vector_2
        
        # Test that conjugate_transpose equals conjugate of transpose
        from ct_laboratory.linear_system.transpose import TransposeLinearSystem
        from ct_laboratory.linear_system.conjugate import ConjugateLinearSystem
        
        transpose_op = TransposeLinearSystem(base_matrix_operator=base_op)
        conjugate_of_transpose = ConjugateLinearSystem(base_matrix_operator=transpose_op)
        
        y1 = op.forward(x)
        y2 = conjugate_of_transpose.forward(x)
        assert torch.allclose(y1, y2)
        
        # Test that conjugate_transpose equals transpose of conjugate
        conjugate_op = ConjugateLinearSystem(base_matrix_operator=base_op)
        transpose_of_conjugate = TransposeLinearSystem(base_matrix_operator=conjugate_op)
        
        y3 = transpose_of_conjugate.forward(x)
        assert torch.allclose(y1, y3) 