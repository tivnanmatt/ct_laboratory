"""
Tests for ct_laboratory.linalg (the consolidated LinearSystem stack ported
from gmi). gmi shipped no linalg tests, so these are written fresh to prove
the merged area works: forward/transpose/inverse contracts, adjoint
consistency, and unitarity of the Fourier operator.
"""
import pytest
import torch

from ct_laboratory.linalg import (
    LinearSystem,
    Identity,
    Scalar,
    DiagonalScalar,
    FourierTransform,
    CompositeLinearSystem,
)


def _adjoint_ok(A, x, y, atol=1e-5):
    """<A x, y> == <x, Aᵀ y> for real tensors."""
    lhs = torch.sum(A.forward(x) * y)
    rhs = torch.sum(x * A.transpose(y))
    return torch.allclose(lhs, rhs, atol=atol)


class TestIdentity:
    def test_forward_is_passthrough(self):
        x = torch.randn(2, 3, 4)
        assert torch.allclose(Identity().forward(x), x)

    def test_transpose_and_inverse_passthrough(self):
        x = torch.randn(5)
        I = Identity()
        assert torch.allclose(I.transpose(x), x)
        assert torch.allclose(I.inverse(x), x)


class TestScalar:
    def test_forward_scales(self):
        x = torch.randn(4, 4)
        assert torch.allclose(Scalar(3.0).forward(x), 3.0 * x)

    def test_inverse_roundtrip(self):
        x = torch.randn(4, 4)
        A = Scalar(2.5)
        assert torch.allclose(A.inverse(A.forward(x)), x, atol=1e-6)

    def test_symmetric_adjoint(self):
        x, y = torch.randn(3, 3), torch.randn(3, 3)
        assert _adjoint_ok(Scalar(1.7), x, y)

    def test_mat_add(self):
        x = torch.randn(4)
        summed = Scalar(2.0).mat_add(Scalar(3.0))
        assert torch.allclose(summed.forward(x), 5.0 * x, atol=1e-6)


class TestDiagonalScalar:
    def test_forward_elementwise(self):
        d = torch.tensor([1.0, 2.0, 3.0])
        x = torch.randn(3)
        assert torch.allclose(DiagonalScalar(d).forward(x), d * x)

    def test_inverse_roundtrip(self):
        d = torch.tensor([1.0, 2.0, 4.0])
        A = DiagonalScalar(d)
        x = torch.randn(3)
        assert torch.allclose(A.inverse(A.forward(x)), x, atol=1e-6)

    def test_adjoint(self):
        d = torch.tensor([0.5, -1.0, 2.0])
        x, y = torch.randn(3), torch.randn(3)
        assert _adjoint_ok(DiagonalScalar(d), x, y)


class TestFourierTransform:
    def test_unitary_roundtrip(self):
        x = torch.randn(8, 8, dtype=torch.complex64)
        F = FourierTransform(dim=(-2, -1))
        recon = F.conjugate_transpose(F.forward(x))
        assert torch.allclose(recon, x, atol=1e-4)

    def test_energy_preserved(self):
        x = torch.randn(16, dtype=torch.complex64)
        F = FourierTransform(dim=(-1,))
        e_in = torch.sum(torch.abs(x) ** 2)
        e_out = torch.sum(torch.abs(F.forward(x)) ** 2)
        assert torch.allclose(e_in, e_out, rtol=1e-4)


class TestComposite:
    def test_composition_order(self):
        # Composite([A, B]) applies as A(B(x)) per gmi convention; check against
        # explicit application with two scalars (commuting, so order-safe here).
        x = torch.randn(4)
        C = CompositeLinearSystem([Scalar(2.0), Scalar(3.0)])
        assert torch.allclose(C.forward(x), 6.0 * x, atol=1e-6)


def test_linear_system_base_forward_not_implemented():
    # gmi's base LinearSystem.forward returns (does not raise) NotImplementedError;
    # concrete subclasses override it. Pin the actual contract.
    assert LinearSystem().forward(torch.randn(3)) is NotImplementedError
