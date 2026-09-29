# File: voxel_projector_3d_autograd.py
"""Autograd function for the voxel-driven separable-footprint projector.

Forward = SF forward projection; backward = SF back projection of the incoming
gradient.  Because both directions use exactly the same per-voxel weights
(recomputed, or read from the same CSR arrays), the backward is the exact
adjoint of the forward.
"""
from __future__ import annotations

import torch

from .voxel_projector_3d_cuda import (
    sf_forward_project_3d_cuda,
    sf_back_project_3d_cuda,
    sf_forward_project_3d_csr_cuda,
    sf_back_project_3d_csr_cuda,
    sf_forward_project_3d_col_cuda,
    sf_back_project_3d_col_cuda,
)
from .voxel_projector_3d_torch import (
    sf_forward_project_3d_torch,
    sf_back_project_3d_torch,
    sf_forward_project_3d_csr_torch,
    sf_back_project_3d_csr_torch,
)

_OPS = {
    "cuda": (sf_forward_project_3d_cuda, sf_back_project_3d_cuda,
             sf_forward_project_3d_csr_cuda, sf_back_project_3d_csr_cuda),
    "torch": (sf_forward_project_3d_torch, sf_back_project_3d_torch,
              sf_forward_project_3d_csr_torch, sf_back_project_3d_csr_torch),
}

__all__ = ["VoxelProjector3DFunction"]


class VoxelProjector3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, M, b, views, view_off, valid, fp_ptr, fp_idx, fp_w, n_pix, backend="cuda",
                col_ptr=None, col_view=None, col_trap=None):
        """volume ``[X,Y,Z]`` or ``[B,X,Y,Z]``.  If ``fp_ptr`` is not None the
        precomputed CSR footprints are used, otherwise footprints are computed
        on the fly."""
        ctx.volume_shape = volume.shape
        ctx.n_pix = int(n_pix)
        ctx.use_csr = fp_ptr is not None
        ctx.backend = backend
        fwd, _, fwd_csr, _ = _OPS[backend]
        if ctx.use_csr:
            ctx.save_for_backward(fp_ptr, fp_idx, fp_w)
            return fwd_csr(volume, fp_ptr, fp_idx, fp_w, n_pix)
        ctx.use_col = col_ptr is not None
        if ctx.use_col:                                   # CUDA only
            ctx.save_for_backward(M, b, views, view_off, valid, col_ptr, col_view, col_trap)
            return sf_forward_project_3d_col_cuda(volume, M, b, views, view_off, valid, col_ptr, col_view, col_trap)
        ctx.save_for_backward(M, b, views, view_off, valid)
        return fwd(volume, M, b, views, view_off, valid)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        shp = ctx.volume_shape
        n_x, n_y, n_z = (shp if len(shp) == 3 else shp[1:])
        _, back, _, back_csr = _OPS[ctx.backend]
        if ctx.use_csr:
            fp_ptr, fp_idx, fp_w = ctx.saved_tensors
            grad_volume = back_csr(grad_output, fp_ptr, fp_idx, fp_w, n_x, n_y, n_z)
        elif getattr(ctx, "use_col", False):
            M, b, views, view_off, valid, col_ptr, col_view, col_trap = ctx.saved_tensors
            grad_volume = sf_back_project_3d_col_cuda(grad_output, M, b, views, view_off, valid,
                                                      col_ptr, col_view, col_trap, n_x, n_y, n_z)
        else:
            M, b, views, view_off, valid = ctx.saved_tensors
            grad_volume = back(grad_output, M, b, views, view_off, valid, n_x, n_y, n_z)
        # grad w.r.t. volume only; geometry / footprints are not differentiable here
        return grad_volume, None, None, None, None, None, None, None, None, None, None, None, None, None
