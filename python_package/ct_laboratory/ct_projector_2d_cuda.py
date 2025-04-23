# File: python_package/ct_laboratory/ct_projector_2d_cuda.py

import torch
import ct_laboratory._C as _C  # your compiled extension module

def compute_intersections_2d_cuda(
    n_row: int,
    n_col: int,
    M: torch.Tensor,      # [2,2]
    b: torch.Tensor,      # [2]
    src: torch.Tensor,    # [n_ray,2]
    dst: torch.Tensor,    # [n_ray,2]
):
    """
    Returns a [n_ray, (n_row+1 + n_col+1)] tensor of sorted t-values in [0,1].
    """
    return _C.compute_intersections_2d(n_row, n_col, src, dst, M, b)


def forward_project_2d_cuda(
    image: torch.Tensor,   # [B,R,C] or [R,C]
    tvals: torch.Tensor,   # [n_ray, n_intersections]
    M: torch.Tensor,       # [2,2]
    b: torch.Tensor,       # [2]
    src: torch.Tensor,     # [n_ray,2]
    dst: torch.Tensor,     # [n_ray,2]
):
    """
    Forward projector using precomputed intersections `tvals`.
    Returns [B, n_ray] if image is [B,R,C], else [n_ray].
    """
    out = _C.forward_project_2d_cuda(image, tvals, src, dst, M, b)
    if image.dim() == 2:  # squeeze if no batch
        out = out.squeeze(0)
    return out


def back_project_2d_cuda(
    sinogram: torch.Tensor,  # [B,n_ray] or [n_ray]
    tvals: torch.Tensor,     # [n_ray, n_intersections]
    M: torch.Tensor,         # [2,2]
    b: torch.Tensor,         # [2]
    src: torch.Tensor,       # [n_ray,2]
    dst: torch.Tensor,       # [n_ray,2]
    n_row: int,
    n_col: int
):
    """
    Back projector using precomputed intersections `tvals`.
    Returns [B,R,C] or [R,C].
    """
    out = _C.back_project_2d_cuda(
        sinogram, tvals, src, dst, M, b,
        n_row, n_col
    )
    # if sinogram was 1D => shape is [1, R, C], so squeeze
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out