import torch
import pytest
from ct_laboratory.linear_system import FourierFilter, EigenDecompositionLinearSystem

class TestFourierFilter:
    def test_instantiation(self):
        filt = torch.ones(8, 8)
        op = FourierFilter(filt, dim=(-2, -1))
        assert isinstance(op, FourierFilter)
        assert torch.allclose(op.filter, filt)
        assert op.dim == (-2, -1)

    def test_forward(self):
        filt = torch.ones(8, 8)
        op = FourierFilter(filt, dim=(-2, -1))
        x = torch.randn(1, 8, 8)
        y = op.forward(x)
        assert y.shape == x.shape
        assert torch.is_complex(y)

    def test_mat_add(self):
        filt1 = torch.ones(8, 8)
        filt2 = 2 * torch.ones(8, 8)
        op1 = FourierFilter(filt1, dim=(-2, -1))
        op2 = FourierFilter(filt2, dim=(-2, -1))
        op_sum = op1.mat_add(op2)
        assert torch.allclose(op_sum.filter, filt1 + filt2)

    def test_mat_sub(self):
        filt1 = 3 * torch.ones(8, 8)
        filt2 = torch.ones(8, 8)
        op1 = FourierFilter(filt1, dim=(-2, -1))
        op2 = FourierFilter(filt2, dim=(-2, -1))
        op_sub = op1.mat_sub(op2)
        assert torch.allclose(op_sub.filter, filt1 - filt2)

    def test_mat_mul(self):
        filt1 = 2 * torch.ones(8, 8)
        filt2 = 3 * torch.ones(8, 8)
        op1 = FourierFilter(filt1, dim=(-2, -1))
        op2 = FourierFilter(filt2, dim=(-2, -1))
        op_mul = op1.mat_mul(op2)
        assert torch.allclose(op_mul.filter, filt1 * filt2)

    def test_inheritance(self):
        from ct_laboratory.linear_system import EigenDecompositionLinearSystem, LinearSystem
        filt = torch.ones(8, 8)
        op = FourierFilter(filt, dim=(-2, -1))
        assert isinstance(op, EigenDecompositionLinearSystem)
        assert isinstance(op, LinearSystem)

    def test_transpose_LinearSystem(self):
        """Test the inherited transpose_LinearSystem method."""
        filt = torch.ones(8, 8)
        op = FourierFilter(filt, dim=(-2, -1))
        transpose_op = op.transpose_LinearSystem()
        
        assert isinstance(transpose_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that transpose operator works correctly
        x = torch.randn(1, 8, 8)
        y_original = op.forward(x)
        y_transpose = transpose_op.forward(x)
        assert torch.allclose(y_original, y_transpose, atol=1e-6), "Transpose operator should give same result as original for symmetric filter"

    def test_conjugate_LinearSystem(self):
        """Test the inherited conjugate_LinearSystem method."""
        filt = torch.ones(8, 8)
        op = FourierFilter(filt, dim=(-2, -1))
        conjugate_op = op.conjugate_LinearSystem()
        
        assert isinstance(conjugate_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that conjugate operator works correctly
        x = torch.randn(1, 8, 8)
        y_original = op.forward(x)
        y_conjugate = conjugate_op.forward(x)
        assert torch.allclose(y_original, y_conjugate, atol=1e-6), "Conjugate operator should give same result as original for real filter"

    def test_conjugate_transpose_LinearSystem(self):
        """Test the inherited conjugate_transpose_LinearSystem method."""
        filt = torch.ones(8, 8)
        op = FourierFilter(filt, dim=(-2, -1))
        conj_transpose_op = op.conjugate_transpose_LinearSystem()
        
        assert isinstance(conj_transpose_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that conjugate transpose operator works correctly
        x = torch.randn(1, 8, 8)
        y_original = op.forward(x)
        y_conj_transpose = conj_transpose_op.forward(x)
        assert torch.allclose(y_original, y_conj_transpose, atol=1e-6), "Conjugate transpose operator should give same result as original for symmetric filter"

    def test_inverse_LinearSystem(self):
        """Test the inherited inverse_LinearSystem method."""
        filt = torch.ones(8, 8) * 2.0  # Non-zero filter
        op = FourierFilter(filt, dim=(-2, -1))
        inverse_op = op.inverse_LinearSystem()
        
        assert isinstance(inverse_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that inverse operator works correctly
        x = torch.randn(1, 8, 8, dtype=torch.complex64)
        y_original = op.forward(x)
        y_inverse = inverse_op.forward(y_original)
        assert torch.allclose(x, y_inverse, atol=1e-6), "Inverse operator should recover original input"

    def test_complex_filter(self):
        """Test with complex filter values."""
        filt = torch.ones(8, 8) * (1.0 + 2.0j)
        op = FourierFilter(filt, dim=(-2, -1))
        
        # Test conjugate
        conjugate_op = op.conjugate_LinearSystem()
        x = torch.randn(1, 8, 8)
        y_original = op.forward(x)
        y_conjugate = conjugate_op.forward(x)
        expected_conjugate = torch.conj(y_original)
        assert torch.allclose(y_conjugate, expected_conjugate, atol=1e-6), "Conjugate should conjugate complex filter"
        
        # Test conjugate transpose
        conj_transpose_op = op.conjugate_transpose_LinearSystem()
        y_conj_transpose = conj_transpose_op.forward(x)
        expected_conj_transpose = torch.conj(y_original)
        assert torch.allclose(y_conj_transpose, expected_conj_transpose, atol=1e-6), "Conjugate transpose should conjugate complex filter"

    def test_operator_chain_operations(self):
        """Test chaining of the inherited LinearSystem methods."""
        filt = torch.ones(8, 8) * 2.0
        op = FourierFilter(filt, dim=(-2, -1))
        x = torch.randn(1, 8, 8)
        
        # Test transpose -> inverse
        transpose_op = op.transpose_LinearSystem()
        inv_transpose_op = transpose_op.inverse_LinearSystem()
        assert isinstance(inv_transpose_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test inverse -> transpose
        inv_op = op.inverse_LinearSystem()
        transpose_inv_op = inv_op.transpose_LinearSystem()
        assert isinstance(transpose_inv_op, EigenDecompositionLinearSystem), "Should return EigenDecompositionLinearSystem"
        
        # Test that operations commute correctly
        y1 = inv_transpose_op.forward(x)
        y2 = transpose_inv_op.forward(x)
        assert torch.allclose(y1, y2, atol=1e-6), "Inverse of transpose should equal transpose of inverse" 