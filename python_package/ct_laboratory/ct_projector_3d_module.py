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
    Computes the 3D intersections in __init__ and stores them.
    Then uses the CTProjector3DFunction for forward/backward passes.
    """
    def __init__(self, n_x, n_y, n_z, M, b, src, dst, backend='torch', device=None):
        """
        n_x, n_y, n_z: volume shape
        M, b: 3D transform (3x3, 3x1)
        src, dst: [n_ray, 3]
        backend: 'torch' or 'cuda'
        """
        
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backend = backend
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z

        # Register geometry
        self.register_buffer('M', M)
        self.register_buffer('b', b)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)

        # Precompute intersections
        if backend == 'torch':
            tvals = compute_intersections_3d_torch(n_x, n_y, n_z, M, b, src, dst)
        else:
            tvals = compute_intersections_3d_cuda(n_x, n_y, n_z, M, b, src, dst)

        # tvals is shape [n_rays, n_itersection]
        # get me a list of the shape [n_rays] with the number of finite intersections per ray
        n_intersections = tvals.shape[1]
        n_rays = tvals.shape[0]
        for i in range(n_intersections):
            current_intersection_list = tvals[:,i]
            # if none are finite, crop tvals there and exit

            if torch.all(torch.isinf(current_intersection_list)):
                _tvals_old = tvals
                tvals = tvals.narrow(1, 0, i)
                del _tvals_old
                gc.collect()
                torch.cuda.empty_cache()
                break
        self.register_buffer('tvals', tvals)

    def forward(self, volume):
        """
        volume: [n_x,n_y,n_z] or [B,n_x,n_y,n_z]
        returns sinogram: [n_ray] or [B,n_ray]
        """
        return CTProjector3DFunction.apply(
            volume, self.tvals, self.M, self.b, self.src, self.dst, self.backend
        )
