#!/usr/bin/env python

import torch
from ct_laboratory.ct_projector_2d_module import CTProjector2DModule

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


def test_ct_projector_2d_module(backend='torch'):
    """
    Build a small 2D phantom, run forward/backward, and verify no errors occur.
    The 'backend' parameter selects either the 'torch' or the custom 'cuda' extension.
    """
    n_row, n_col = 16, 16
    # Create phantom without requires_grad first
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    phantom[8, :] = 1.0
    # Mark phantom as requiring grad AFTER modification to avoid in-place ops on a leaf.
    phantom.requires_grad_()

    M = torch.eye(2, dtype=torch.float32)
    b = torch.zeros(2, dtype=torch.float32)
    
    src = torch.tensor([
        [8.0, -5.0],
        [0.0,  0.0],
        [15.0, 15.0],
    ], dtype=torch.float32)
    dst = torch.tensor([
        [8.0, 25.0],
        [15.0, 15.0],
        [0.0,  8.0],
    ], dtype=torch.float32)

    # Use the same device for all tensors regardless of backend choice.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom = phantom.to(device)
    M = M.to(device)
    b = b.to(device)
    src = src.to(device)
    dst = dst.to(device)

    projector = CTProjector2DModule(
        n_row=n_row,
        n_col=n_col,
        M=M,
        b=b,
        src=src,
        dst=dst,
        backend=backend
    ).to(device)

    sinogram = projector(phantom)
    print(f"[{backend}] 2D sinogram shape = {sinogram.shape}")

    loss = sinogram.sum()
    loss.backward()

    grad_ok = phantom.grad is not None
    print(f"[{backend}] 2D test => grad_ok = {grad_ok}")
    print(f"[{backend}] 2D test => Completed forward/backward without error!")

if __name__ == "__main__":
    for bk in ['torch', 'cuda']:
        try:
            test_ct_projector_2d_module(backend=bk)
        except Exception as e:
            print(f"Error testing backend={bk}: {e}")
