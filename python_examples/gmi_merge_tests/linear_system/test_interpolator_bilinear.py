import torch
import pytest
from ct_laboratory.linear_system import BilinearInterpolator

def test_bilinear_interpolator():
    num_row, num_col = 4, 4
    # Simple grid: 0 1 2 3 in each row
    x = torch.arange(num_row * num_col, dtype=torch.float32).reshape(1, 1, num_row, num_col)
    # Interpolate at (1.5, 1.5) should be the average of 4 neighbors: 5, 6, 9, 10
    interp_points = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
    interp = BilinearInterpolator(num_row, num_col, interp_points)
    result = interp.forward(x)
    expected = torch.tensor([(5 + 6 + 9 + 10) / 4], dtype=torch.float32).view(1, 1, -1)
    assert torch.allclose(result, expected, atol=1e-5) 