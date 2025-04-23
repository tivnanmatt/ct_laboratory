#!/usr/bin/env python
import math
import torch
import matplotlib.pyplot as plt

# Import the conebeam projector module from the new file.
from ct_laboratory.conebeam_projector_3d import ConeBeam3DProjector

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")

def build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=30.0):
    """Creates a 3D volume with a sphere inside."""
    vol = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    cx = (n_x - 1) / 2.0 + center[0]
    cy = (n_y - 1) / 2.0 + center[1]
    cz = (n_z - 1) / 2.0 + center[2]
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                if (i - cx)**2 + (j - cy)**2 + (k - cz)**2 < radius**2:
                    vol[i, j, k] = 1.0
    return vol

def main():
    # Define volume dimensions and cone-beam geometry parameters.
    n_x, n_y, n_z = 64, 64, 64
    n_view = 120
    det_nx, det_ny = 100, 100
    sid = 200.0  # Source-to-Isocenter Distance
    sdd = 400.0  # Source-to-Detector Distance
    det_spacing = 1.0
    voxel_size = 1.0

    # Build a 3D spherical phantom.
    volume = build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=20.0)

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume = volume.to(device)

    # Instantiate the ConeBeam3DProjector.
    projector = ConeBeam3DProjector(
        n_x, n_y, n_z,
        n_view, det_nx, det_ny,
        sid, sdd,
        det_spacing=det_spacing,
        voxel_size=voxel_size,
        backend="cuda"
    )
    projector.to(device)

    # Perform forward projection.
    with torch.no_grad():
        sinogram = projector(volume)

    # Reshape the sinogram to [n_view, det_nx, det_ny] and display one view.
    sino_reshaped = sinogram.view(n_view, det_nx, det_ny).cpu().numpy()
    plt.imshow(sino_reshaped[0], cmap='gray', origin='lower')
    plt.title("Cone-Beam Sinogram (View 0)")
    plt.xlabel("Detector U")
    plt.ylabel("Detector V")
    
    
    # save the plot
    plt.savefig(f"{output_dir}/conebeam_sinogram.png")

if __name__ == "__main__":
    main()
