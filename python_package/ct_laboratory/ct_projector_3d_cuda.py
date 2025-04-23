# File: ct_projector_3d_cuda.py
import torch
import ct_laboratory._C as _C  # Your compiled extension module

def compute_intersections_3d_cuda(
    n_x: int,
    n_y: int,
    n_z: int,
    M: torch.Tensor,      # [3,3]
    b: torch.Tensor,      # [3]
    src: torch.Tensor,    # [n_ray,3]
    dst: torch.Tensor,    # [n_ray,3]
):
    """
    Returns [n_ray, (n_x+1 + n_y+1 + n_z+1)] of sorted t in [0,1].
    """
    return _C.compute_intersections_3d(n_x, n_y, n_z, src, dst, M, b)


def forward_project_3d_cuda(
    volume: torch.Tensor,   # [B, n_x, n_y, n_z] or [n_x, n_y, n_z]
    tvals: torch.Tensor,    # [n_ray, n_intersections]
    M: torch.Tensor,        # [3,3]
    b: torch.Tensor,        # [3]
    src: torch.Tensor,      # [n_ray,3]
    dst: torch.Tensor,      # [n_ray,3]
):
    """
    Forward projector using precomputed intersections tvals in 3D.
    Returns [B, n_ray] or [n_ray].
    """
    out = _C.forward_project_3d_cuda(volume, tvals, src, dst, M, b)
    if volume.dim() == 3:
        out = out.squeeze(0)  # remove batch dim
    return out


def back_project_3d_cuda(
    sinogram: torch.Tensor,  # [B, n_ray] or [n_ray]
    tvals: torch.Tensor,     # [n_ray, n_intersections]
    M: torch.Tensor,         # [3,3]
    b: torch.Tensor,         # [3]
    src: torch.Tensor,       # [n_ray,3]
    dst: torch.Tensor,       # [n_ray,3]
    n_x: int,
    n_y: int,
    n_z: int
):
    """
    Back projector using precomputed intersections tvals in 3D.
    Returns [B, n_x, n_y, n_z] or [n_x, n_y, n_z].
    """
    out = _C.back_project_3d_cuda(
        sinogram, tvals, src, dst, M, b,
        n_x, n_y, n_z
    )
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out
