import torch
import numpy as np
import pytest
from ct_laboratory.linear_system import PolarCoordinateResampler

def make_circle_image(num_row, num_col, radius):
    x = torch.zeros(1, 1, num_row, num_col)
    center_row, center_col = num_row // 2, num_col // 2
    for i in range(num_row):
        for j in range(num_col):
            dist = np.sqrt((i - center_row)**2 + (j - center_col)**2)
            if dist < radius:
                x[0, 0, i, j] = 1.0
    return x

@pytest.mark.parametrize("interpolator", ["nearest", "bilinear", "lanczos"])
def test_polar_coordinate_resampler(interpolator):
    num_row, num_col = 32, 32
    theta_values = torch.linspace(0, 2 * np.pi, 16)
    radius_values = torch.linspace(0, 14, 8)
    x = make_circle_image(num_row, num_col, radius=10)

    polar_resampler = PolarCoordinateResampler(
        num_row=num_row,
        num_col=num_col,
        theta_values=theta_values,
        radius_values=radius_values,
        interpolator=interpolator
    )

    # Forward transformation
    result = polar_resampler.forward(x)
    assert result.shape == (1, 1, len(theta_values), len(radius_values))
    assert torch.max(result) <= 1.1 and torch.min(result) >= -0.1

    # Transpose transformation
    back_result = polar_resampler.transpose(result)
    assert back_result.shape == (1, 1, num_row, num_col)
    # The back-projection should have nonzero values in the circle region
    assert torch.max(back_result) > 0.5
    # Should be mostly zero outside the circle
    assert torch.min(back_result) >= -0.5 