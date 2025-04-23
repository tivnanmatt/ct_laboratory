#!/usr/bin/env python

import torch
from ct_laboratory.ct_projector_3d_module import CTProjector3DModule

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


def test_ct_projector_3d_module(backend='torch'):
    """
    Build a small 3D volume, run forward/backward, and verify no errors occur.
    The 'backend' parameter selects either the 'torch' or the custom 'cuda' extension.
    """
    n_x, n_y, n_z = 8, 8, 8
    # Create volume without requires_grad first, then set requires_grad
    volume = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    volume[4, :, :] = 1.0
    volume.requires_grad_()

    M = torch.eye(3, dtype=torch.float32)
    b = torch.zeros(3, dtype=torch.float32)

    src = torch.tensor([
        [4.0, -5.0, 4.0],
        [0.0,  0.0, 0.0],
    ], dtype=torch.float32)
    dst = torch.tensor([
        [4.0, 15.0, 4.0],
        [7.0,  7.0, 7.0],
    ], dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume = volume.to(device)
    M = M.to(device)
    b = b.to(device)
    src = src.to(device)
    dst = dst.to(device)

    projector = CTProjector3DModule(
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        M=M,
        b=b,
        src=src,
        dst=dst,
        backend=backend
    ).to(device)

    sinogram = projector(volume)
    print(f"[{backend}] 3D sinogram shape = {sinogram.shape}")

    loss = sinogram.sum()
    loss.backward()

    grad_ok = volume.grad is not None
    print(f"[{backend}] 3D test => grad_ok = {grad_ok}")
    print(f"[{backend}] 3D test => Completed forward/backward without error!")

if __name__ == "__main__":
    for bk in ['torch', 'cuda']:
        try:
            test_ct_projector_3d_module(backend=bk)
        except Exception as e:
            print(f"Error testing backend={bk}: {e}")
