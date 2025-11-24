import torch
from torch import nn
from .ct_projector_3d_module import CTProjector3DModule
from .ct_projector_3d_multirot import MultiRotationProjector
import gc


class RandomSubsetProjector(nn.Module):

    def __init__(self, base_projector, backend='torch', device=None, precomputed_intersections=False, subset_size=10000, randomize_on_forward=False, randomize_on_backward=False, seed=None):
        super().__init__()
        assert isinstance(base_projector, CTProjector3DModule)
        assert not base_projector.precomputed_intersections, "Base projector must not use precomputed intersections."
        assert not isinstance(base_projector, MultiRotationProjector), "Base projector must not be a MultiRotationProjector."
        self.base_projector = base_projector
        self.n_x = base_projector.n_x
        self.n_y = base_projector.n_y
        self.n_z = base_projector.n_z
        self.M = base_projector.M
        self.b = base_projector.b
        self.backend = backend
        self.device = device
        self.precomputed_intersections = precomputed_intersections
        self.subset_size = subset_size
        self.seed = seed
        self.randomize_on_forward = randomize_on_forward
        self.randomize_on_backward = randomize_on_backward
        self.random_subset_projector = None
        self.randomize_projector()
    
    def randomize_projector(self, subset_size=None, seed=None):
        if self.random_subset_projector is not None:
            del self.random_subset_projector
            gc.collect()
            # also clean gpu memory
            if self.device is not None and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize(self.device)
        if subset_size is not None: 
            self.subset_size = subset_size
            
        if self.seed is None:
            seed = self.seed
        assert hasattr(self, 'base_projector'), "Base projector not found."
        assert isinstance(self.base_projector, CTProjector3DModule)
        self.n_rays = self.base_projector.src.shape[0]
        assert self.base_projector.src.shape[0] == self.base_projector.dst.shape[0], "Source and destination must have the same number of rays."
        # get random indices without replacement
        # apply seed first
        if seed is not None:
            torch.manual_seed(seed)

        self.random_subset_indices = torch.randperm(self.n_rays, device='cpu')[:self.subset_size].to(self.device)

        self.src = self.base_projector.src[self.random_subset_indices]
        self.dst = self.base_projector.dst[self.random_subset_indices]
        self.random_subset_projector = CTProjector3DModule(
            n_x=self.n_x,
            n_y=self.n_y,
            n_z=self.n_z,
            M=self.M,
            b=self.b,
            src=self.src,
            dst=self.dst,
            backend=self.backend,
            device=self.device,
            precomputed_intersections=self.precomputed_intersections
        )

        return 

    def forward_project(self, volume):
        return self.random_subset_projector.forward_project(volume)
    
    def back_project(self, sinogram):
        return self.random_subset_projector.back_project(sinogram)
    
    def forward(self, volume):
        """
        volume: [n_x,n_y,n_z] or [B,n_x,n_y,n_z]
        returns sinogram: [n_ray] or [B,n_ray]
        """
        return self.random_subset_projector.forward(volume)