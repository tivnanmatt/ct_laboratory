import torch

# For Torch-based forward/back:
from .ct_projector_2d_torch import (
    forward_project_2d_torch,
    back_project_2d_torch
)

# For CUDA-based forward/back:
from .ct_projector_2d_cuda import (
    forward_project_2d_cuda,
    back_project_2d_cuda
)


class CTProjector2DFunction(torch.autograd.Function):
    """
    This autograd Function performs a 2D forward projection on the forward pass
    and a 2D back projection on the backward pass, using precomputed intersections.
    We allow either the 'torch' or 'cuda' backend.
    """
    @staticmethod
    def forward(ctx, image, tvals, M, b, src, dst, backend='torch'):
        """
        image: 2D or 3D Tensor [R,C] or [B,R,C]
        tvals: precomputed intersection params [n_ray, n_intersections]
        M    : [2,2]
        b    : [2]
        src, dst: [n_ray, 2]
        backend: 'torch' or 'cuda'
        Returns sinogram: [n_ray] or [B, n_ray]
        """
        # Save the needed tensors for backward
        ctx.save_for_backward(tvals, M, b, src, dst)
        ctx.backend = backend
        ctx.image_shape = image.shape  # Store shape for the backward dimension logic

        # Forward projection
        if backend == 'torch':
            sinogram = forward_project_2d_torch(image, tvals, M, b, src, dst)
        else:  # 'cuda'
            sinogram = forward_project_2d_cuda(image, tvals, M, b, src, dst)
        return sinogram

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: same shape as the sinogram: [n_ray] or [B, n_ray]
        We compute d(L)/d(image) by back-projecting grad_output.
        """
        tvals, M, b, src, dst = ctx.saved_tensors
        backend = ctx.backend
        image_shape = ctx.image_shape

        # Determine shape for back-projection
        if len(image_shape) == 2:
            n_row, n_col = image_shape
        else:  # shape = [B, R, C]
            _, n_row, n_col = image_shape

        # Back projection => gradient w.r.t. image
        if backend == 'torch':
            grad_image = back_project_2d_torch(
                grad_output, tvals, M, b, src, dst, n_row, n_col
            )
        else:
            grad_image = back_project_2d_cuda(
                grad_output, tvals, M, b, src, dst, n_row, n_col
            )

        # Return gradient for `image` only; the rest get None
        return grad_image, None, None, None, None, None, None
