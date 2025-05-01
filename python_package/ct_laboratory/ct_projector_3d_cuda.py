# File: ct_projector_3d_cuda.py
"""GPU‑accelerated 3‑D projector / back‑projector utilities
----------------------------------------------------------------
This module exposes three public families of functions that the higher‑level
code (modules + autograd functions) rely on:

1. **Pre‑computed intersections**
   * ``compute_intersections_3d_cuda``
   * ``forward_project_3d_cuda``
   * ``back_project_3d_cuda``

2. **On‑the‑fly Siddon traversal** (no intersection list)
   * ``forward_project_3d_on_the_fly_cuda``
   * ``back_project_3d_on_the_fly_cuda``

``forward_project_3d_cuda`` / ``back_project_3d_cuda`` now accept **``tvals`` that
may be ``None`` or an *empty tensor*.  When ``tvals`` is missing they fall back to
the Siddon kernels so that the call‑site (autograd) does **not** have to change
its signature.
"""

from __future__ import annotations

import torch
import ct_laboratory._C as _C  # C++/CUDA extension

__all__ = [
    "compute_intersections_3d_cuda",
    "forward_project_3d_cuda",
    "back_project_3d_cuda",
    "forward_project_3d_on_the_fly_cuda",
    "back_project_3d_on_the_fly_cuda",
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _prepare_M_inv(M: torch.Tensor) -> torch.Tensor:
    """Return **row‑major** 3×3 inverse of ``M`` as ``float32``, contiguous on GPU."""
    if not M.is_cuda:
        raise TypeError("M must be a CUDA tensor (got CPU)")
    return torch.inverse(M).to(dtype=torch.float32, copy=False).contiguous()


def _empty_or_none(t: torch.Tensor | None) -> bool:  # utility for flexible API
    return (t is None) or (t.numel() == 0)

# -----------------------------------------------------------------------------
# 1. Pre‑computed intersection utilities
# -----------------------------------------------------------------------------

def compute_intersections_3d_cuda(
    n_x: int,
    n_y: int,
    n_z: int,
    M: torch.Tensor,
    b: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Return *sorted* parametric intersections (**t**) of every ray.

    Shape: ``[n_ray, (n_x+1)+(n_y+1)+(n_z+1)]``
    """
    
    M_inv = _prepare_M_inv(M)
    return _C.compute_intersections_3d(n_x, n_y, n_z, src, dst, M_inv, b)

# ----------------------------------------------------
# Forward projector (pre‑computed or Siddon fallback)
# ----------------------------------------------------

def forward_project_3d_cuda(
    volume: torch.Tensor,
    tvals: torch.Tensor | None,
    M: torch.Tensor,
    b: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Line‐integral forward projector.

    Parameters
    ----------
    volume : ``[B, X, Y, Z]`` or ``[X, Y, Z]``
    tvals  : pre‑computed intersections **or** *None* / empty to use Siddon
    M, b   : affine that maps (i,j,k) → (x,y,z)
    src,dst: ``[n_ray, 3]`` world‑space coordinates
    """

    assert volume.is_contiguous(), "Volume must be C-contiguous"
    assert volume.dtype == torch.float32, "Volume must be float32"
    assert src.is_contiguous(), "src must be C-contiguous"
    assert src.dtype == torch.float32, "src must be float32"
    assert dst.is_contiguous(), "dst must be C-contiguous"
    assert dst.dtype == torch.float32, "dst must be float32"
    assert M.is_contiguous(), "M must be C-contiguous"
    assert M.dtype == torch.float32, "M must be float32"
    assert b.is_contiguous(), "b must be C-contiguous"
    assert b.dtype == torch.float32, "b must be float32"

    if not _empty_or_none(tvals):
        assert tvals.is_contiguous(), "tvals must be C-contiguous"
        assert tvals.dtype == torch.float32, "tvals must be float32"

        M_inv = _prepare_M_inv(M)

        out = _C.forward_project_3d_cuda(volume, tvals, src, dst, M_inv, b)

        if volume.dim() == 3:
            out = out.squeeze(0)
        return out

    else:
        return forward_project_3d_on_the_fly_cuda(volume, M, b, src, dst)

    

# ----------------------------------------------------
# Back‑projector (pre‑computed or Siddon fallback)
# ----------------------------------------------------

def back_project_3d_cuda(
    sinogram: torch.Tensor,
    tvals: torch.Tensor | None,
    M: torch.Tensor,
    b: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_x: int,
    n_y: int,
    n_z: int,
) -> torch.Tensor:
    
    # --- NEW: enforce dtype + contiguity on all inputs ---
    # 1) Volume must be [B,X,Y,Z] or [X,Y,Z], float32, C-order
    # if not volume.is_contiguous() or volume.dtype != torch.float32:
        # volume = volume.contiguous().to(dtype=torch.float32)
    assert sinogram.is_contiguous(), "Volume must be C-contiguous"
    assert sinogram.dtype == torch.float32, "Volume must be float32"

    # 2) tvals must be float32 & contiguous (if used)
    if not _empty_or_none(tvals):
        # if tvals.dtype != torch.float32 or not tvals.is_contiguous():
        #     tvals = tvals.contiguous().to(dtype=torch.float32)
        assert tvals.is_contiguous(), "tvals must be C-contiguous"
        assert tvals.dtype == torch.float32, "tvals must be float32"

    # 3) src/dst must be [n_ray,3], float32, contiguous
    # if src.dtype != torch.float32 or not src.is_contiguous():
    #     src = src.contiguous().to(dtype=torch.float32)
    assert src.is_contiguous(), "src must be C-contiguous"
    assert src.dtype == torch.float32, "src must be float32"

    # if dst.dtype != torch.float32 or not dst.is_contiguous():
    #     dst = dst.contiguous().to(dtype=torch.float32)
    assert dst.is_contiguous(), "dst must be C-contiguous"
    assert dst.dtype == torch.float32, "dst must be float32"

    """Adjoint of the forward projector with optional Siddon fallback."""
    if _empty_or_none(tvals):
        return back_project_3d_on_the_fly_cuda(sinogram, M, b, src, dst, n_x, n_y, n_z)

    M_inv = _prepare_M_inv(M)
    out = _C.back_project_3d_cuda(sinogram, tvals, src, dst, M_inv, b, n_x, n_y, n_z)
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out

# -----------------------------------------------------------------------------
# 2. On‑the‑fly Siddon wrappers (thin convenience layers)
# -----------------------------------------------------------------------------

def forward_project_3d_on_the_fly_cuda(
    volume: torch.Tensor,
    M: torch.Tensor,
    b: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """Siddon forward projector (no intersection list)."""

    
    assert volume.is_contiguous(), "Volume must be C-contiguous"
    assert volume.dtype == torch.float32, "Volume must be float32"
    assert src.is_contiguous(), "src must be C-contiguous"
    assert src.dtype == torch.float32, "src must be float32"
    assert dst.is_contiguous(), "dst must be C-contiguous"
    assert dst.dtype == torch.float32, "dst must be float32"
    assert M.is_contiguous(), "M must be C-contiguous"
    assert M.dtype == torch.float32, "M must be float32"
    assert b.is_contiguous(), "b must be C-contiguous"
    assert b.dtype == torch.float32, "b must be float32"


    M_inv = _prepare_M_inv(M)
    out = _C.forward_project_3d_on_the_fly_cuda(volume, src, dst, M_inv, b)
    if volume.dim() == 3:
        out = out.squeeze(0)
    return out


def back_project_3d_on_the_fly_cuda(
    sinogram: torch.Tensor,
    M: torch.Tensor,
    b: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    n_x: int,
    n_y: int,
    n_z: int,
) -> torch.Tensor:
    """Siddon back projector (no intersection list)."""

    
    assert sinogram.is_contiguous(), "Volume must be C-contiguous"
    assert sinogram.dtype == torch.float32, "Volume must be float32"
    assert src.is_contiguous(), "src must be C-contiguous"
    assert src.dtype == torch.float32, "src must be float32"
    assert dst.is_contiguous(), "dst must be C-contiguous"
    assert dst.dtype == torch.float32, "dst must be float32"
    assert M.is_contiguous(), "M must be C-contiguous"
    assert M.dtype == torch.float32, "M must be float32"
    assert b.is_contiguous(), "b must be C-contiguous"
    assert b.dtype == torch.float32, "b must be float32"


    M_inv = _prepare_M_inv(M)
    out = _C.back_project_3d_on_the_fly_cuda(sinogram, src, dst, M_inv, b, n_x, n_y, n_z)
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out
