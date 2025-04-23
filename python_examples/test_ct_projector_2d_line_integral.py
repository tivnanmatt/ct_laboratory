#!/usr/bin/env python

import time
import torch

# 1) Torch version
from ct_laboratory.ct_projector_2d_torch import (
    compute_intersections_2d_torch,
    forward_project_2d_torch
)

# 2) CUDA version
from ct_laboratory.ct_projector_2d_cuda import (
    compute_intersections_2d_cuda,
    forward_project_2d_cuda
)

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")



def test_line_integral_2d(backend="torch"):
    """
    Compute the line integral of a single ray passing through a 2D image.
    No visualization, only prints results.
    """

    # 1) Build a 32x32 phantom with a single row of 1.0 at row=16
    n_row, n_col = 32, 32
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    phantom[16, :] = 1.0  # Row 16 is all ones

    # 2) Define the identity transform (row, col) == (x, y)
    A = torch.eye(2, dtype=torch.float32)
    b = torch.zeros(2, dtype=torch.float32)

    # 3) Define ONE ray that passes through row=16
    #    src = (16, -5), dst = (16, 37) => crosses all cols 0..31 in row=16
    src = torch.tensor([[16.0, -5.0]], dtype=torch.float32)  # shape [1,2]
    dst = torch.tensor([[16.0, 37.0]], dtype=torch.float32)  # shape [1,2]

    device = "cuda" if backend == "cuda" else "cpu"
    phantom, A, b, src, dst = map(lambda x: x.to(device), [phantom, A, b, src, dst])

    print(f"\n=== Running single-ray 2D test with backend={backend} ===")

    # 4) Compute intersections ONCE
    t0 = time.perf_counter()
    if backend == "torch":
        tvals = compute_intersections_2d_torch(n_row, n_col, A, b, src, dst)
    else:  # CUDA
        tvals = compute_intersections_2d_cuda(n_row, n_col, A, b, src, dst)
    t1 = time.perf_counter()
    intersection_time = t1 - t0
    print(f"[{backend}] Intersections computed in {intersection_time:.6f} s.")

    # 5) Compute forward projection (line integral)
    t2 = time.perf_counter()
    if backend == "torch":
        sinogram_1d = forward_project_2d_torch(phantom, tvals, A, b, src, dst)
    else:
        sinogram_1d = forward_project_2d_cuda(phantom, tvals, A, b, src, dst)
    t3 = time.perf_counter()
    forward_time = t3 - t2
    print(f"[{backend}] Forward projection (line integral) computed in {forward_time:.6f} s.")

    # 6) Print result: The line integral (should equal 32.0 since all crossed pixels are 1.0)
    print(f"[{backend}] Line integral result = {sinogram_1d.item():.6f}")


if __name__ == "__main__":
    test_line_integral_2d(backend="torch")
    test_line_integral_2d(backend="cuda")
