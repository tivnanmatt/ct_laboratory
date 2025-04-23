#!/usr/bin/env python

import math
import time
import torch
import matplotlib.pyplot as plt

# 1) Torch version
from ct_laboratory.ct_projector_2d_torch import (
    compute_intersections_2d_torch,
    forward_project_2d_torch,
    back_project_2d_torch
)

# 2) CUDA version
from ct_laboratory.ct_projector_2d_cuda import (
    compute_intersections_2d_cuda,
    forward_project_2d_cuda,
    back_project_2d_cuda
)

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
            dist2 = (row - row_center)**2 + (col - col_center)**2
            if dist2 < radius**2:
                phantom[row, col] = 1.0
    return phantom


def build_fanbeam_rays(center_xy, n_view, n_det, source_distance, detector_distance, det_spacing):
    """Generates source & detector positions for a 2D fan-beam CT system."""
    cx, cy = center_xy
    ds, dd = source_distance, detector_distance
    angles = torch.arange(0, n_view) * (2 * math.pi / n_view)

    all_src, all_dst = [], []
    for theta in angles:
        sx, sy = cx + ds * math.cos(theta), cy + ds * math.sin(theta)
        dx_center, dy_center = cx - dd * math.cos(theta), cy - dd * math.sin(theta)

        # Perpendicular vector to the ray
        perp_x, perp_y = -(dy_center - sy), (dx_center - sx)
        norm_len = math.sqrt(perp_x**2 + perp_y**2)
        if norm_len < 1e-12:
            continue
        perp_x /= norm_len
        perp_y /= norm_len

        mid_i = (n_det - 1) / 2.0
        for i in range(n_det):
            offset = (i - mid_i) * det_spacing
            cell_x = dx_center + offset * perp_x
            cell_y = dy_center + offset * perp_y

            all_src.append([sx, sy])
            all_dst.append([cell_x, cell_y])

    return torch.tensor(all_src, dtype=torch.float32), torch.tensor(all_dst, dtype=torch.float32)
    return torch.tensor(all_src, dtype=torch.float32), torch.tensor(all_dst, dtype=torch.float32)


def run_fanbeam_experiment(
    n_row=256, n_col=256, n_view=360, n_det=400, ds=200.0, dd=200.0, det_spacing=1.0,
    backend="torch", out_prefix="fanbeam", show_plot=False
):
    """Run a fan-beam forward & back projection demo with either the Torch or CUDA backend."""

    # 1) Build 2D circular phantom
    phantom = build_circular_phantom(n_row, n_col, center_offset=(-20, 10), radius=50.0)

    # 2) Transform: (row,col) -> (x,y) coordinates
    A = torch.eye(2, dtype=torch.float32)
    row_mid, col_mid = (n_row - 1) / 2.0, (n_col - 1) / 2.0
    b = torch.tensor([-row_mid, -col_mid], dtype=torch.float32)

    # 3) Build fanbeam geometry
    src, dst = build_fanbeam_rays(
        center_xy=(0, 0), n_view=n_view, n_det=n_det,
        source_distance=ds, detector_distance=dd, det_spacing=det_spacing
    )
    total_rays = src.shape[0]

    device = "cuda" if backend == "cuda" else "cpu"
    phantom, A, b, src, dst = map(lambda x: x.to(device), [phantom, A, b, src, dst])

    print(f"\n=== Running fanbeam experiment with backend={backend} ===")

    # 4) Compute intersections ONCE
    t0 = time.perf_counter()
    if backend == "torch":
        tvals = compute_intersections_2d_torch(n_row, n_col, A, b, src, dst)
    else:  # CUDA
        tvals = compute_intersections_2d_cuda(n_row, n_col, A, b, src, dst)
    t1 = time.perf_counter()
    intersection_time = t1 - t0
    print(f"[{backend}] Intersections computed in {intersection_time:.4f} s.")

    # 5) Forward projection
    t2 = time.perf_counter()
    if backend == "torch":
        sinogram_1d = forward_project_2d_torch(phantom, tvals, A, b, src, dst)
    else:  # CUDA
        sinogram_1d = forward_project_2d_cuda(phantom, tvals, A, b, src, dst)
    t3 = time.perf_counter()
    forward_time = t3 - t2
    print(f"[{backend}] Forward projection completed in {forward_time:.4f} s.")

    # 6) Back projection
    t4 = time.perf_counter()
    if backend == "torch":
        reco = back_project_2d_torch(sinogram_1d, tvals, A, b, src, dst, n_row, n_col)
    else:  # CUDA
        reco = back_project_2d_cuda(sinogram_1d, tvals, A, b, src, dst, n_row, n_col)
    t5 = time.perf_counter()
    backward_time = t5 - t4
    print(f"[{backend}] Back projection completed in {backward_time:.4f} s.")

    # Convert to CPU for plotting
    phantom_cpu = phantom.cpu().numpy()
    sinogram_cpu = sinogram_1d.view(n_view, n_det).cpu().numpy()
    reco_cpu = reco.cpu().numpy()

    # 7) Save & display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(phantom_cpu, origin='lower', cmap='gray')
    axs[0].set_title("Original Phantom")

    axs[1].imshow(sinogram_cpu, aspect='auto', cmap='gray')
    axs[1].set_title("Fan-Beam Sinogram")

    axs[2].imshow(reco_cpu, origin='lower', cmap='gray')
    axs[2].set_title("Back-Projected Image")

    for ax in axs:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.tight_layout()
    out_filename = f"{output_dir}/{out_prefix}_{backend}.png"
    plt.savefig(out_filename, dpi=200)
    print(f"Saved figure => {out_filename}")

    if show_plot:
        plt.show()

    return intersection_time, forward_time, backward_time


if __name__ == "__main__":
    for backend in ["torch", "cuda"]:
        i_time, f_time, b_time = run_fanbeam_experiment(backend=backend, show_plot=True)
        print(f"\n=== Timing Summary ({backend.upper()}) ===")
        print(f"Intersection Time: {i_time:.4f} s")
        print(f"Forward Projection Time: {f_time:.4f} s")
        print(f"Backward Projection Time: {b_time:.4f} s")
