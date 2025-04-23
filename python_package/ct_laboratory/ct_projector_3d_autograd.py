import torch

# For Torch-based forward/back:
from .ct_projector_3d_torch import (
    forward_project_3d_torch,
    back_project_3d_torch
)

# For CUDA-based forward/back:
from .ct_projector_3d_cuda import (
    forward_project_3d_cuda,
    back_project_3d_cuda
)


class CTProjector3DFunction(torch.autograd.Function):
    """
    Autograd Function for 3D forward/back projection with precomputed intersections.
    Supports either 'torch' or 'cuda' backend.
    """
    @staticmethod
    def forward(ctx, volume, tvals, M, b, src, dst, backend='torch'):
        """
        volume: [n_x,n_y,n_z] or [B,n_x,n_y,n_z]
        tvals:  [n_ray, n_intersections]
        M:      [3,3]
        b:      [3]
        src,dst:[n_ray,3]
        backend:'torch' or 'cuda'
        Returns sinogram: [n_ray] or [B, n_ray]
        """
        ctx.save_for_backward(tvals, M, b, src, dst)
        ctx.backend = backend
        ctx.volume_shape = volume.shape

        if backend == 'torch':
            sinogram = forward_project_3d_torch(volume, tvals, M, b, src, dst)
        else:
            sinogram = forward_project_3d_cuda(volume, tvals, M, b, src, dst)

        return sinogram

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [n_ray] or [B,n_ray]
        Returns grad_volume with same shape as input volume.
        """
        tvals, M, b, src, dst = ctx.saved_tensors
        backend = ctx.backend
        volume_shape = ctx.volume_shape

        if len(volume_shape) == 3:
            n_x, n_y, n_z = volume_shape
        else:
            # volume_shape = [B, n_x, n_y, n_z]
            _, n_x, n_y, n_z = volume_shape

        if backend == 'torch':
            grad_volume = back_project_3d_torch(
                grad_output, tvals, M, b, src, dst, n_x, n_y, n_z
            )
        else:
            grad_volume = back_project_3d_cuda(
                grad_output, tvals, M, b, src, dst, n_x, n_y, n_z
            )

        # Return gradient w.r.t. volume only
        return grad_volume, None, None, None, None, None, None
