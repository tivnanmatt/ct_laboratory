import torch

# Intersection computations for Torch or CUDA
from .ct_projector_2d_torch import compute_intersections_2d_torch
from .ct_projector_2d_cuda import compute_intersections_2d_cuda

# Our new autograd Function
from .ct_projector_2d_autograd import CTProjector2DFunction


class CTProjector2DModule(torch.nn.Module):
    """
    Computes the 2D intersections in __init__ and stores them.
    Then uses the CTProjector2DFunction for forward/backward passes.
    """
    def __init__(self, n_row, n_col, M, b, src, dst, backend='torch'):
        """
        n_row, n_col: image shape
        M, b: 2D transform
        src, dst: [n_ray,2]
        backend: 'torch' or 'cuda'
        """
        super().__init__()
        self.backend = backend
        self.n_row = n_row
        self.n_col = n_col

        # Register M,b,src,dst as buffers so they move with .cuda()/to(device)
        self.register_buffer('M', M)
        self.register_buffer('b', b)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)

        # Precompute intersections
        if backend == 'torch':
            tvals = compute_intersections_2d_torch(n_row, n_col, M, b, src, dst)
        else:
            tvals = compute_intersections_2d_cuda(n_row, n_col, M, b, src, dst)

        # Store as a buffer
        self.register_buffer('tvals', tvals)

    def forward(self, image):
        """
        image: [R,C] or [B,R,C]
        Returns sinogram: same batch dimension or none, shape => [n_ray] or [B,n_ray].
        """
        return CTProjector2DFunction.apply(
            image, self.tvals, self.M, self.b, self.src, self.dst, self.backend
        )
