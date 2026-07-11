import torch
import pytest
from ct_laboratory.linear_system import NearestNeighborInterpolator

def test_nearest_neighbor_interpolator():
    num_row, num_col = 4, 4
    # Simple grid: 0 1 2 3 in each row
    x = torch.arange(num_row * num_col, dtype=torch.float32).reshape(1, 1, num_row, num_col)
    # Interpolate at exact grid points
    interp_points = torch.tensor([
        [0, 0], [1, 1], [2, 2], [3, 3], [0, 3], [3, 0]
    ], dtype=torch.float32)
    interp = NearestNeighborInterpolator(num_row, num_col, interp_points)
    result = interp.forward(x)
    # Should match the values at those points
    expected = torch.tensor([0, 5, 10, 15, 3, 12], dtype=torch.float32).view(1, 1, -1)
    assert torch.allclose(result, expected, atol=1e-5) 