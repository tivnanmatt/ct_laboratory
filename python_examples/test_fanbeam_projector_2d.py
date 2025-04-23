#!/usr/bin/env python
import math
import torch
import matplotlib.pyplot as plt

# Import the fanbeam projector module from the new file.
from ct_laboratory.fanbeam_projector_2d import FanBeam2DProjector

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


def build_circular_phantom(n_row, n_col, center_offset=(-20, 10), radius=50.0):
    """Creates a 2D circular phantom in a [n_row, n_col] image."""
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    row_center = (n_row - 1) / 2.0 + center_offset[0]
    col_center = (n_col - 1) / 2.0 + center_offset[1]
    for row in range(n_row):
        for col in range(n_col):
            if (row - row_center)**2 + (col - col_center)**2 < radius**2:
                phantom[row, col] = 1.0
    return phantom

def main():
    # Define image dimensions and fan-beam geometry parameters.
    n_row, n_col = 256, 256
    n_view = 360
    n_det = 400
    sid = 200.0  # Source-to-Isocenter Distance
    sdd = 400.0  # Source-to-Detector Distance
    det_spacing = 1.0
    pixel_size = 1.0

    # Build phantom image.
    phantom = build_circular_phantom(n_row, n_col)

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom = phantom.to(device)

    # Instantiate the FanBeam2DProjector.
    projector = FanBeam2DProjector(
        n_row, n_col,
        n_view, n_det,
        sid, sdd,
        det_spacing=det_spacing,
        pixel_size=pixel_size,
        backend="cuda"
    )
    projector.to(device)

    # Perform forward projection.
    with torch.no_grad():
        sinogram = projector(phantom)

    # Reshape and display the sinogram.
    sinogram_image = sinogram.view(n_view, n_det).cpu().numpy()
    plt.imshow(sinogram_image, cmap='gray', aspect='auto')
    plt.title("Fan-Beam Sinogram")
    plt.xlabel("Detector Element")
    plt.ylabel("Projection Angle")
    # save the plot
    plt.savefig(f"{output_dir}/fanbeam_sinogram.png")
if __name__ == "__main__":
    main()