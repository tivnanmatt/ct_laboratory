import torch
import pytest
from ct_laboratory.linear_system import ColSparseLinearSystem

def make_identity_col_sparse(input_shape):
    # For a 2x2 image, flatten indices: 0 1 2 3
    # Each output pixel gets its corresponding input pixel
    indices = torch.tensor([
        [[0, 1], [2, 3]]  # shape [1, 2, 2]
    ])
    weights = torch.ones_like(indices, dtype=torch.float32)
    return ColSparseLinearSystem(input_shape, input_shape, indices, weights)

@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_col_sparse_linear_operator(dtype):
    op = make_identity_col_sparse((2, 2))
    x = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    if dtype.is_complex:
        x = x.to(dtype)
    y = op.forward(x)
    assert y.shape == (1, 1, 2, 2)
    # Should act like identity for this setup
    assert torch.allclose(y, x, atol=1e-5)

    # Test transpose
    y_t = op.transpose(y)
    assert y_t.shape == (1, 1, 2, 2)
    assert torch.allclose(y_t, x, atol=1e-5)

    # Test conjugate (should be same for real input)
    y_conj = op.conjugate(x)
    assert torch.allclose(y_conj, y, atol=1e-5)

    # Test conjugate transpose
    y_conj_t = op.conjugate_transpose(y)
    assert torch.allclose(y_conj_t, x, atol=1e-5) 