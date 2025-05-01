import torch

# Intersection computations
from .ct_projector_3d_torch import compute_intersections_3d_torch
from .ct_projector_3d_cuda import compute_intersections_3d_cuda

# Autograd function
from .ct_projector_3d_autograd import CTProjector3DFunction

# garbage collection
import gc


class CTProjector3DModule(torch.nn.Module):
    """
    CT projector module supporting precomputed or on-the-fly Siddon projection.
    """
    def __init__(self, n_x, n_y, n_z, M, b, src, dst,
                 backend='torch', device=None, precomputed_intersections=False):
        """
        n_x, n_y, n_z: volume shape
        M, b: 3D transform (3x3, 3x1)
        src, dst: [n_ray, 3]
        backend: 'torch' or 'cuda'
        precomputed_intersections: if True, uses precomputed Siddon t-values
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backend = backend
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.precomputed_intersections = precomputed_intersections

        # Register geometry buffers
        self.register_buffer('M', M)
        self.register_buffer('b', b)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)

        # Either precompute or defer to on-the-fly Siddon
        if precomputed_intersections:
            if backend == 'torch':
                tvals = compute_intersections_3d_torch(n_x, n_y, n_z, M, b, src, dst)
            else:
                tvals = compute_intersections_3d_cuda(n_x, n_y, n_z, M, b, src, dst)

            # Trim trailing INFINITY-only columns
            n_intersections = tvals.shape[1]
            for i in range(n_intersections):
                if torch.all(torch.isinf(tvals[:, i])):
                    tvals = tvals.narrow(1, 0, i)
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
            tvals = tvals.contiguous()
            self.register_buffer('tvals', tvals)
        else:
            self.register_buffer('tvals', None)

    def forward(self, volume):
        """
        volume: [n_x,n_y,n_z] or [B,n_x,n_y,n_z]
        returns sinogram: [n_ray] or [B,n_ray]
        """
        return CTProjector3DFunction.apply(
            volume, self.tvals, self.M, self.b, self.src, self.dst, self.backend
        )
