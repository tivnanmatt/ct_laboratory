#!/usr/bin/env python

import math
import time
import torch
import matplotlib.pyplot as plt

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


# 1) Torch version
from ct_laboratory.ct_projector_3d_torch import (
    compute_intersections_3d_torch,
    forward_project_3d_torch,
    back_project_3d_torch
)

# 2) CUDA version
from ct_laboratory.ct_projector_3d_cuda import (
    compute_intersections_3d_cuda,
    forward_project_3d_cuda,
    back_project_3d_cuda
)


def build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=30.0):
    """Creates a 3D volume with a sphere inside."""
    vol = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    cx, cy, cz = (n_x - 1) / 2 + center[0], (n_y - 1) / 2 + center[1], (n_z - 1) / 2 + center[2]

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                dist2 = (i - cx)**2 + (j - cy)**2 + (k - cz)**2
                if dist2 <= radius**2:
                    vol[i, j, k] = 1.0
    return vol
def build_conebeam_geometry_3d(
    n_x, n_y, n_z,  # Image volume dimensions
    n_view=90, det_nx=64, det_ny=64, ds=200.0, dd=200.0, det_spacing=1.0
):
    """
    Generates a 3D cone-beam projection geometry rotating around the **center of the volume**.
    
    Assumes voxel indices are directly in world coordinates (A=I, b=0).
    """
    all_src, all_dst = [], []
    
    # Compute center of the volume
    cx, cy, cz = (n_x - 1) / 2, (n_y - 1) / 2, (n_z - 1) / 2

    # Define rotation angles
    angles = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_view, n_view)

    for theta in angles:
        # Source position (rotating around volume center)
        sx = cx + ds * math.cos(theta)
        sy = cy + ds * math.sin(theta)
        sz = cz  # Keep z-source centered

        # Detector center position
        dx_c = cx - dd * math.cos(theta)
        dy_c = cy - dd * math.sin(theta)
        dz_c = cz  # Keep detector centered in z

        # Define detector plane axes
        perp_x, perp_y = -(dy_c - sy), (dx_c - sx)
        norm_len = math.sqrt(perp_x**2 + perp_y**2)
        if norm_len < 1e-9:
            continue
        perp_x /= norm_len
        perp_y /= norm_len
        v_x, v_y, v_z = 0.0, 0.0, 1.0  # Vertical axis (Z)

        # Detector grid
        mid_u, mid_v = (det_nx - 1) / 2, (det_ny - 1) / 2
        for iu in range(det_nx):
            for iv in range(det_ny):
                offset_u = (iu - mid_u) * det_spacing
                offset_v = (iv - mid_v) * det_spacing
                cell_x = dx_c + offset_u * perp_x + offset_v * v_x
                cell_y = dy_c + offset_u * perp_y + offset_v * v_y
                cell_z = dz_c + offset_u * 0.0 + offset_v * v_z

                all_src.append([sx, sy, sz])
                all_dst.append([cell_x, cell_y, cell_z])

    return torch.tensor(all_src, dtype=torch.float32), torch.tensor(all_dst, dtype=torch.float32)


def run_conebeam_experiment(
    n_x=64, n_y=64, n_z=64, n_view=90, det_nx=100, det_ny=100, ds=200.0, dd=200.0, det_spacing=1.0,
    backend="torch", out_prefix="conebeam", show_plot=False
):
    """Run a cone-beam forward & back projection demo with either the Torch or CUDA backend."""

    # 1) Build 3D sphere volume
    volume = build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=20.0)

    # 2) Transform: (i,j,k) -> (x,y,z)
    A = torch.eye(3, dtype=torch.float32)
    # b = torch.tensor([(n_x - 1) / 2, (n_y - 1) / 2, (n_z - 1) / 2], dtype=torch.float32)
    b = torch.zeros(3, dtype=torch.float32)

    # 3) Build cone-beam projection geometry
    src, dst = build_conebeam_geometry_3d(n_x, n_y, n_z, n_view, det_nx, det_ny, ds, dd, det_spacing)
    total_rays = src.shape[0]

    # device = "cuda" if backend == "cuda" else "cpu"
    device = "cuda"
    volume, A, b, src, dst = map(lambda x: x.to(device), [volume, A, b, src, dst])

    print(f"\n=== Running cone-beam experiment with backend={backend} ===")

    # 4) Compute intersections ONCE
    t0 = time.perf_counter()
    if backend == "torch":
        tvals = compute_intersections_3d_torch(n_x, n_y, n_z, A, b, src, dst)
    else:  # CUDA
        tvals = compute_intersections_3d_cuda(n_x, n_y, n_z, A, b, src, dst)
    t1 = time.perf_counter()
    intersection_time = t1 - t0
    print(f"[{backend}] Intersections computed in {intersection_time:.4f} s.")

    # 5) Forward projection
    t2 = time.perf_counter()
    if backend == "torch":
        sinogram_1d = forward_project_3d_torch(volume, tvals, A, b, src, dst)
    else:  # CUDA
        sinogram_1d = forward_project_3d_cuda(volume, tvals, A, b, src, dst)
    t3 = time.perf_counter()
    forward_time = t3 - t2
    print(f"[{backend}] Forward projection completed in {forward_time:.4f} s.")

    # 6) Back projection
    t4 = time.perf_counter()
    if backend == "torch":
        reco = back_project_3d_torch(sinogram_1d, tvals, A, b, src, dst, n_x, n_y, n_z)
    else:  # CUDA
        reco = back_project_3d_cuda(sinogram_1d, tvals, A, b, src, dst, n_x, n_y, n_z)
    t5 = time.perf_counter()
    backward_time = t5 - t4
    print(f"[{backend}] Back projection completed in {backward_time:.4f} s.")

    # Convert to CPU for plotting
    volume_cpu = volume.cpu().numpy()
    sino_2d = sinogram_1d.view(n_view, det_nx, det_ny).cpu().numpy()
    reco_cpu = reco.cpu().numpy()
    mid_z = n_z // 2

    # 7) Save & display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(volume_cpu[:, :, mid_z], cmap='gray', origin='lower')
    axs[0].set_title("Original Sphere (mid slice)")

    axs[1].imshow(sino_2d[0], cmap='gray')
    axs[1].set_title("Cone-Beam Sinogram (view=0)")

    axs[2].imshow(reco_cpu[:, :, mid_z], cmap='gray', origin='lower')
    axs[2].set_title("Back-Projection (mid slice)")

    plt.tight_layout()
    out_filename = f"{output_dir}/{out_prefix}_{backend}.png"
    plt.savefig(out_filename, dpi=150)
    print(f"Saved figure => {out_filename}")

    if show_plot:
        plt.show()

    return intersection_time, forward_time, backward_time


if __name__ == "__main__":
    for backend in ["torch", "cuda"]:
        i_time, f_time, b_time = run_conebeam_experiment(backend=backend, show_plot=True)
        print(f"\n=== Timing Summary ({backend.upper()}) ===")
        print(f"Intersection Time: {i_time:.4f} s")
        print(f"Forward Projection Time: {f_time:.4f} s")
        print(f"Backward Projection Time: {b_time:.4f} s")
