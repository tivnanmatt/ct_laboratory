# File: voxel_projector_3d_cuda.py
"""Thin, checked wrappers around the voxel-driven separable-footprint CUDA kernels.

Two execution modes, bit-identical in their weights:

* **on-the-fly** – footprints recomputed inside every forward/back call
  (``sf_forward_project_3d_cuda`` / ``sf_back_project_3d_cuda``)
* **CSR**        – footprints precomputed once into per-voxel lists of
  (pixel index, weight) (``sf_precompute_footprints_3d_cuda``) and then applied
  with ``sf_forward_project_3d_csr_cuda`` / ``sf_back_project_3d_csr_cuda``.

Conventions (identical to the ray-driven code): volume ``[X,Y,Z]`` or
``[B,X,Y,Z]`` C-contiguous float32 with Z fastest; ``world = M @ (i,j,k) + b``.
Sinogram is flat, length ``n_pix``; see :mod:`voxel_projector_3d_module` for
the view/pixel ordering.
"""
from __future__ import annotations

import torch
import ct_laboratory._C as _C

__all__ = [
    "sf_forward_project_3d_cuda",
    "sf_back_project_3d_cuda",
    "sf_precompute_footprints_3d_cuda",
    "sf_forward_project_3d_csr_cuda",
    "sf_back_project_3d_csr_cuda",
    "sf_precompute_columns_3d_cuda",
    "sf_forward_project_3d_col_cuda",
    "sf_back_project_3d_col_cuda",
    "VIEW_STRIDE",
]

VIEW_STRIDE = 19   # floats per view record, see sf_projector_3d.cu


def _f32c(t: torch.Tensor, name: str) -> torch.Tensor:
    if not t.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    return t.to(dtype=torch.float32).contiguous()


def _geometry_args(M, b, views, view_off, valid):
    if M.dim() > 2:
        M = M[0]
    if b.dim() > 1:
        b = b[0]
    M = _f32c(M, "M").reshape(3, 3)
    b = _f32c(b, "b").reshape(3)
    views = _f32c(views, "views")
    assert views.dim() == 2 and views.shape[1] == VIEW_STRIDE, "views must be [n_view, 19]"
    view_off = view_off.to(device=views.device, dtype=torch.int64).contiguous()
    assert view_off.numel() == views.shape[0] + 1, "view_off must be [n_view + 1]"
    if valid is None or valid.numel() == 0:
        valid = torch.empty(0, dtype=torch.uint8, device=views.device)
    else:
        valid = valid.to(device=views.device, dtype=torch.uint8).contiguous().reshape(-1)
        assert valid.numel() == int(view_off[-1]), "valid must have n_pix entries"
    return M, b, views, view_off, valid


# ---------------------------------------------------------------------------
# on-the-fly
# ---------------------------------------------------------------------------
def sf_forward_project_3d_cuda(volume, M, b, views, view_off, valid=None):
    """Voxel-driven SF forward projection (footprints computed on the fly)."""
    volume = _f32c(volume, "volume")
    M, b, views, view_off, valid = _geometry_args(M, b, views, view_off, valid)
    out = _C.sf_forward_project_3d_cuda(volume, M, b, views, view_off, valid)
    if volume.dim() == 3:
        out = out.squeeze(0)
    return out


def sf_back_project_3d_cuda(sinogram, M, b, views, view_off, valid, n_x, n_y, n_z):
    """Voxel-driven SF back projection (exact adjoint, no atomics)."""
    sinogram = _f32c(sinogram, "sinogram")
    M, b, views, view_off, valid = _geometry_args(M, b, views, view_off, valid)
    out = _C.sf_back_project_3d_cuda(sinogram, M, b, views, view_off, valid, n_x, n_y, n_z)
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# precomputed per-voxel footprints (CSR)
# ---------------------------------------------------------------------------
def sf_precompute_footprints_3d_cuda(n_x, n_y, n_z, M, b, views, view_off, valid=None):
    """Return ``(ptr[int64, n_vox+1], idx[int32, nnz], w[float32, nnz])``.

    Voxel ``v`` (flat index ``i*n_y*n_z + j*n_z + k``) touches sinogram pixels
    ``idx[ptr[v]:ptr[v+1]]`` with weights ``w[ptr[v]:ptr[v+1]]`` (mm).
    """
    M, b, views, view_off, valid = _geometry_args(M, b, views, view_off, valid)
    return _C.sf_precompute_footprints_3d_cuda(n_x, n_y, n_z, M, b, views, view_off, valid)


def sf_forward_project_3d_csr_cuda(volume, ptr, idx, w, n_pix):
    volume = _f32c(volume, "volume")
    out = _C.sf_forward_project_3d_csr_cuda(volume, ptr, idx, w, int(n_pix))
    if volume.dim() == 3:
        out = out.squeeze(0)
    return out


def sf_back_project_3d_csr_cuda(sinogram, ptr, idx, w, n_x, n_y, n_z):
    sinogram = _f32c(sinogram, "sinogram")
    out = _C.sf_back_project_3d_csr_cuda(sinogram, ptr, idx, w, n_x, n_y, n_z)
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out


# ---------------------------------------------------------------------------
# column cache: per transaxial column, culled view list + stored trapezoid
# ---------------------------------------------------------------------------
def sf_precompute_columns_3d_cuda(n_x, n_y, n_z, M, b, views, view_off, valid):
    """Returns ``(col_ptr int64 [n_x*n_y+1], col_view int32 [E], col_trap float32 [E,8])``."""
    M, b, views, view_off, valid = _geometry_args(M, b, views, view_off, valid)
    return _C.sf_precompute_columns_3d_cuda(int(n_x), int(n_y), int(n_z), M, b, views, view_off, valid)


def sf_forward_project_3d_col_cuda(volume, M, b, views, view_off, valid, col_ptr, col_view, col_trap):
    M, b, views, view_off, valid = _geometry_args(M, b, views, view_off, valid)
    volume = _f32c(volume, "volume")
    out = _C.sf_forward_project_3d_col_cuda(volume, M, b, views, view_off, valid,
                                            col_ptr.contiguous(), col_view.contiguous(), col_trap.contiguous())
    return out.squeeze(0) if volume.dim() == 3 else out


def sf_back_project_3d_col_cuda(sino, M, b, views, view_off, valid, col_ptr, col_view, col_trap, n_x, n_y, n_z):
    M, b, views, view_off, valid = _geometry_args(M, b, views, view_off, valid)
    sino = _f32c(sino, "sino")
    out = _C.sf_back_project_3d_col_cuda(sino, M, b, views, view_off, valid,
                                         col_ptr.contiguous(), col_view.contiguous(), col_trap.contiguous(),
                                         int(n_x), int(n_y), int(n_z))
    return out.squeeze(0) if sino.dim() == 1 else out
