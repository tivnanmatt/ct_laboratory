import torch
import pytest
from ct_laboratory.linear_system import LanczosInterpolator

def test_lanczos_interpolator():
    num_row, num_col = 4, 4
    # Simple grid: 0 1 2 3 in each row
    x = torch.arange(num_row * num_col, dtype=torch.float32).reshape(1, 1, num_row, num_col)
    # Interpolate at (1.5, 1.5) should be a weighted sum of neighbors
    interp_points = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
    interp = LanczosInterpolator(num_row, num_col, interp_points, kernel_size=3)
    result = interp.forward(x)
    # Should be between min and max of the 3x3 neighborhood
    min_val = x[0, 0, 1:3, 1:3].min()
    max_val = x[0, 0, 1:3, 1:3].max()
    assert (result >= min_val - 1e-3).all() and (result <= max_val + 1e-3).all()
    # Weights for the single point should sum to 1
    weights_sum = interp.weights[:, 0].sum()
    assert torch.allclose(weights_sum, torch.tensor(1.0), atol=1e-3) 