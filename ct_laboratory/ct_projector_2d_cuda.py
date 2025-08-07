# File: python_package/ct_laboratory/ct_projector_2d_cuda.py
import torch
import ct_laboratory._C as _C      # re-compile after updating .cu!

# ------------------------------------------------------------------
def _M_inv(M: torch.Tensor) -> torch.Tensor:
    """Return contiguous float32 inverse on the same device as M."""
    return torch.inverse(M).to(dtype=torch.float32).contiguous()

# ------------------------------------------------------------------
def compute_intersections_2d_cuda(
    n_row: int, n_col: int,
    M: torch.Tensor, b: torch.Tensor,
    src: torch.Tensor, dst: torch.Tensor):
    """[n_ray, (n_row+1 + n_col+1)] of sorted t-values."""
    return _C.compute_intersections_2d(
        n_row, n_col, src, dst, _M_inv(M), b
    )

# ------------------------------------------------------------------
def forward_project_2d_cuda(
    image: torch.Tensor, tvals: torch.Tensor,
    M: torch.Tensor, b: torch.Tensor,
    src: torch.Tensor, dst: torch.Tensor):
    """Return [B,n_ray] or [n_ray] line integrals."""
    out = _C.forward_project_2d_cuda(
        image, tvals, src, dst, _M_inv(M), b
    )
    if image.dim() == 2:
        out = out.squeeze(0)
    return out

# ------------------------------------------------------------------
def back_project_2d_cuda(
    sinogram: torch.Tensor, tvals: torch.Tensor,
    M: torch.Tensor, b: torch.Tensor,
    src: torch.Tensor, dst: torch.Tensor,
    n_row: int, n_col: int):
    """Return [B,R,C] or [R,C] back-projection."""
    out = _C.back_project_2d_cuda(
        sinogram, tvals, src, dst, _M_inv(M), b,
        n_row, n_col
    )
    if sinogram.dim() == 1:
        out = out.squeeze(0)
    return out
