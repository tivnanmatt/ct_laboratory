# File: ct_projector_3d_cuda.py
import torch
import ct_laboratory._C as _C       # compiled extension (re-compiled with new .cu)

# ---------------------------------------------------------------------
# helper – compute inverse once and keep contiguous
def _prepare_M_inv(M: torch.Tensor) -> torch.Tensor:
    # user passes M on any device / dtype - keep in float32
    return torch.inverse(M).to(dtype=torch.float32).contiguous()

# ---------------------------------------------------------------------
def compute_intersections_3d_cuda(
    n_x: int, n_y: int, n_z: int,
    M: torch.Tensor, b: torch.Tensor,
    src: torch.Tensor, dst: torch.Tensor):
    """
    Returns [n_ray, n_intersections] sorted t-values.  We invert M once
    on the Python side and pass the inverse down to the CUDA kernels.
    """
    M_inv = _prepare_M_inv(M)
    return _C.compute_intersections_3d(
        n_x, n_y, n_z, src, dst, M_inv, b
    )

# ---------------------------------------------------------------------
def forward_project_3d_cuda(
    volume: torch.Tensor,   # [B, n_x, n_y, n_z]  or [n_x, n_y, n_z]
    tvals: torch.Tensor,    # [n_ray, n_intersections]
    M: torch.Tensor, b: torch.Tensor,
    src: torch.Tensor, dst: torch.Tensor):
    """
    Forward projector (CUDA).  Returns [B, n_ray] or [n_ray].
    """
    M_inv = _prepare_M_inv(M)
    out = _C.forward_project_3d_cuda(
        volume, tvals, src, dst, M_inv, b
    )
    if volume.dim() == 3:
        out = out.squeeze(0)
    return out

# ---------------------------------------------------------------------
def back_project_3d_cuda(
    sinogram: torch.Tensor, tvals: torch.Tensor,
    M: torch.Tensor, b: torch.Tensor,
    src: torch.Tensor, dst: torch.Tensor,
    n_x: int, n_y: int, n_z: int):
    """
    Back-projector (CUDA).  Returns [B, n_x, n_y, n_z] or [n_x, n_y, n_z].
    """
    M_inv = _prepare_M_inv(M)
    out = _C.back_project_3d_cuda(
        sinogram, tvals, src, dst, M_inv, b, n_x, n_y, n_z
    )
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out
