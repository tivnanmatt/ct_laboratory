#!/usr/bin/env python

import time
import torch

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


# 1) Torch version
from ct_laboratory.ct_projector_3d_torch import (
    compute_intersections_3d_torch,
    forward_project_3d_torch,
    back_project_3d_torch,
)

# 2) CUDA version
from ct_laboratory.ct_projector_3d_cuda import (
    compute_intersections_3d_cuda,
    forward_project_3d_cuda,
    back_project_3d_cuda,
)


def test_line_integral_3d(backend="torch"):
    """
    Compute the line integral of a single ray passing through a 3D volume.
    No visualization, only prints results.
    """

    # 1) Build a 32x32x32 volume with a single plane of 1.0 at z=16
    n_x, n_y, n_z = 32, 32, 32
    volume = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    volume[:,16, 16] = 1.0  # Slice z=16 is all ones

    # 2) Define the identity transform (x, y, z) == (i, j, k)
    A = torch.eye(3, dtype=torch.float32)
    b = torch.zeros(3, dtype=torch.float32)

    # 3) Define ONE ray that passes through the center of slice z=16
    #    src = (-5, 16, 16), dst = (37, 16, 16) => crosses all x=0..31 at y=16, z=16
    src = torch.tensor([[200.0, 16.0, 16.0]], dtype=torch.float32)  # shape [1,3]
    dst = torch.tensor([[-200.0, 16.0, 16.0]], dtype=torch.float32)  # shape [1,3]

    device = "cuda" if backend == "cuda" else "cpu"
    volume, A, b, src, dst = map(lambda x: x.to(device), [volume, A, b, src, dst])

    print(f"\n=== Running single-ray 3D test with backend={backend} ===")

    # 4) Compute intersections ONCE
    t0 = time.perf_counter()
    if backend == "torch":
        tvals = compute_intersections_3d_torch(n_x, n_y, n_z, A, b, src, dst)
    else:  # CUDA
        tvals = compute_intersections_3d_cuda(n_x, n_y, n_z, A, b, src, dst)
    t1 = time.perf_counter()
    intersection_time = t1 - t0
    print(f"[{backend}] Intersections computed in {intersection_time:.6f} s.")

    # 5) Compute forward projection (line integral)
    t2 = time.perf_counter()
    if backend == "torch":
        sinogram_1d = forward_project_3d_torch(volume, tvals, A, b, src, dst)
    else:
        sinogram_1d = forward_project_3d_cuda(volume, tvals, A, b, src, dst)
    t3 = time.perf_counter()
    forward_time = t3 - t2
    print(f"[{backend}] Forward projection (line integral) computed in {forward_time:.6f} s.")

    # 6) Print result: The line integral (should equal 32.0 since all crossed voxels are 1.0)
    print(f"[{backend}] Line integral result = {sinogram_1d.item():.6f}")


if __name__ == "__main__":
    test_line_integral_3d(backend="torch")
    test_line_integral_3d(backend="cuda")
