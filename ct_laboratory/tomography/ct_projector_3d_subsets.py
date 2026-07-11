import os
import torch
from torch import nn
import numpy as np
import time
import gc
from .ct_projector_3d_module import CTProjector3DModule

class SubsetProjector(nn.Module):
    """Base class for subset-based projectors."""
    def __init__(self, base_projector, subset_size, precomputed_intersections=False, verbose=False):
        super().__init__()
        self.base_projector = base_projector
        self.subset_size = subset_size
        self.precomputed_intersections = precomputed_intersections
        self.verbose = verbose

        # Resolve down to the innermost CTProjector3DModule
        core = base_projector
        while hasattr(core, 'base_projector') and not isinstance(core, CTProjector3DModule):
            core = core.base_projector
        if not isinstance(core, CTProjector3DModule):
            raise ValueError("base_projector must eventually be a CTProjector3DModule")

        self.core_module = core
        self.n_total_rays_core = core.src.shape[0]

        device = core.src.device
        if device.type == 'cpu':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.n_x = core.n_x
        self.n_y = core.n_y
        self.n_z = core.n_z
        self.M = core.M
        self.b = core.b
        self.backend = core.backend

    def _get_subset_module(self, indices):
        """Create a CTProjector3DModule for a specific subset of core rays (on-the-fly computation)."""
        return CTProjector3DModule(
            n_x=self.n_x, n_y=self.n_y, n_z=self.n_z,
            M=self.M.to(self.device), b=self.b.to(self.device),
            src=self.core_module.src[indices].to(self.device),
            dst=self.core_module.dst[indices].to(self.device),
            backend=self.backend, device=self.device,
            precomputed_intersections=self.precomputed_intersections
        )

    def _run_forward(self, module, volume):
        return module.forward(volume)

    def _run_backward(self, module, sinogram):
        return module.back_project(sinogram)

class OrderedSubsetProjector(SubsetProjector):
    """
    Handles memory limits by processing rays in manageable subsets.
    For each subset, weights are loaded/sliced from the core module (or computed
    and cached to ``cache_dir``), then the subset projector is applied.
    """
    def __init__(self, base_projector, max_subset_size=1_000_000, cache_dir=None, verbose=False):
        super().__init__(base_projector, max_subset_size, precomputed_intersections=True, verbose=verbose)
        self.cache_dir = cache_dir
        self.n_subsets = int(np.ceil(self.n_total_rays_core / self.subset_size))
        self.subset_boundaries = np.linspace(0, self.n_total_rays_core, self.n_subsets + 1, dtype=int)
        self.last_precompute_time = 0.0
        self.last_projection_time = 0.0

        # If everything fits in one subset, persist the precomputed module.
        self.persistent_projector = None
        if self.n_subsets == 1:
            if self.verbose:
                print(f"OrderedSubsetProjector: Single subset ({self.n_total_rays_core} rays). Persisting module.")
            t0 = time.time()
            self.persistent_projector = self._get_cached_subset_projector(0)
            self.persistent_projector.to(self.device)
            self.last_precompute_time = time.time() - t0

    # ── caching helpers ───────────────────────────────────────────────────────

    def _get_subset_cache_path(self, j):
        """Return the on-disk cache path for subset j, or None if no cache_dir."""
        if self.cache_dir is None:
            return None
        start_ray = int(self.subset_boundaries[j])
        end_ray   = int(self.subset_boundaries[j + 1])
        fname = (
            f"tvals_subset{j:04d}_rays{start_ray}-{end_ray}"
            f"_nx{self.n_x}_ny{self.n_y}_nz{self.n_z}.pt"
        )
        return os.path.join(self.cache_dir, fname)

    def _get_cached_subset_projector(self, j):
        """
        Build a CTProjector3DModule for subset j.

        Priority for tvals:
        1. Slice from core_module's already-loaded tvals (fastest, no disk I/O).
        2. Load from per-subset cache file in ``cache_dir``.
        3. Compute with precompute_tvals_stitched and save to ``cache_dir``.
        """
        start_ray = int(self.subset_boundaries[j])
        end_ray   = int(self.subset_boundaries[j + 1])
        core = self.core_module

        if core.tvals_uint16 is not None:
            sub_tvals = (
                core.tvals_uint16[start_ray:end_ray],
                core.tvals_start[start_ray:end_ray],
                core.tvals_scale[start_ray:end_ray],
            )
        elif core.tvals_full is not None:
            sub_tvals = core.tvals_full[start_ray:end_ray]
        else:
            cache_path = self._get_subset_cache_path(j)
            if cache_path is not None and os.path.exists(cache_path):
                if self.verbose:
                    print(f"  Loading subset {j} tvals: {cache_path}", flush=True)
                sub_tvals = torch.load(cache_path, map_location='cpu')
            else:
                from .ct_projector_3d_module import precompute_tvals_stitched
                if self.verbose:
                    print(
                        f"  Computing subset {j} tvals "
                        f"({end_ray - start_ray} rays)...", flush=True
                    )
                sub_tvals = precompute_tvals_stitched(
                    self.n_x, self.n_y, self.n_z,
                    self.M, self.b,
                    core.src[start_ray:end_ray],
                    core.dst[start_ray:end_ray],
                    verbose=self.verbose,
                )
                if cache_path is not None:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    if self.verbose:
                        print(f"  Saving subset {j} tvals → {cache_path}", flush=True)
                    torch.save(sub_tvals, cache_path)

        return CTProjector3DModule(
            n_x=self.n_x, n_y=self.n_y, n_z=self.n_z,
            M=self.M, b=self.b,
            src=core.src[start_ray:end_ray],
            dst=core.dst[start_ray:end_ray],
            backend=self.backend,
            tvals=sub_tvals,
        )
        
    def forward(self, volume):
        self.last_precompute_time = 0.0
        self.last_projection_time = 0.0

        if self.persistent_projector is not None:
            t0 = time.time()
            res = self._run_forward(self.persistent_projector, volume)
            self.last_projection_time = time.time() - t0
            return res

        device = volume.device
        n_rays = self.n_total_rays_core
        if volume.dim() == 4:
            sino_out = torch.zeros((volume.shape[0], n_rays), device=device, dtype=volume.dtype)
        else:
            sino_out = torch.zeros(n_rays, device=device, dtype=volume.dtype)

        for j in range(self.n_subsets):
            start = int(self.subset_boundaries[j])
            end   = int(self.subset_boundaries[j + 1])
            if self.verbose:
                print(f"  [Subset {j+1}/{self.n_subsets}] rays {start}:{end}...", end="", flush=True)
            t0 = time.time()
            t_pre = time.time()
            subset_proj = self._get_cached_subset_projector(j)
            subset_proj.to(device)
            self.last_precompute_time += time.time() - t_pre
            t_proj = time.time()
            sino_chunk = self._run_forward(subset_proj, volume)
            self.last_projection_time += time.time() - t_proj
            if sino_out.dim() == 1:
                sino_out[start:end] = sino_chunk
            else:
                sino_out[:, start:end] = sino_chunk
            del subset_proj
            gc.collect()
            torch.cuda.empty_cache()
            if self.verbose:
                print(f" {time.time() - t0:.2f}s")

        return sino_out

    def back_project(self, sinogram):
        self.last_precompute_time = 0.0
        self.last_projection_time = 0.0

        if self.persistent_projector is not None:
            t0 = time.time()
            res = self._run_backward(self.persistent_projector, sinogram)
            self.last_projection_time = time.time() - t0
            return res

        volume_out = None
        for j in range(self.n_subsets):
            start = int(self.subset_boundaries[j])
            end   = int(self.subset_boundaries[j + 1])
            if sinogram.dim() == 1:
                sino_chunk = sinogram[start:end]
            else:
                sino_chunk = sinogram[:, start:end]
            if self.verbose:
                print(
                    f"  [Subset {j+1}/{self.n_subsets}] rays {start}:{end} (backward)...",
                    end="", flush=True,
                )
            t0 = time.time()
            t_pre = time.time()
            subset_proj = self._get_cached_subset_projector(j)
            subset_proj.to(sinogram.device)
            self.last_precompute_time += time.time() - t_pre
            t_proj = time.time()
            vol_chunk = self._run_backward(subset_proj, sino_chunk)
            self.last_projection_time += time.time() - t_proj
            if volume_out is None:
                volume_out = vol_chunk
            else:
                volume_out += vol_chunk
            del subset_proj
            gc.collect()
            torch.cuda.empty_cache()
            if self.verbose:
                print(f" {time.time() - t0:.2f}s")

        return volume_out

class RandomSubsetProjector(SubsetProjector):
    """
    Picks a random subset of core rays and applies them across all rotations.
    """
    def __init__(self, base_projector, subset_size=10000, precomputed_intersections=False, randomize_on_forward=False, randomize_on_backward=False, seed=None, verbose=False):
        super().__init__(base_projector, subset_size, precomputed_intersections, verbose)
        self.randomize_on_forward = randomize_on_forward
        self.randomize_on_backward = randomize_on_backward
        self.seed = seed
        self.current_subset_module = None
        self.randomize_projector()

    def randomize_projector(self, subset_size=None, seed=None):
        if subset_size is not None: self.subset_size = subset_size
        if seed is not None: self.seed = seed
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            
        indices = torch.randperm(self.n_total_rays_core, device='cpu')[:self.subset_size]
        
        if self.current_subset_module is not None:
            del self.current_subset_module
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        self.current_subset_module = self._get_subset_module(indices)

    def forward(self, volume):
        if self.randomize_on_forward:
            self.randomize_projector()
        return self._run_forward(self.current_subset_module, volume)

    def back_project(self, sinogram):
        if self.randomize_on_backward:
            self.randomize_projector()
        return self._run_backward(self.current_subset_module, sinogram)

    def forward_project(self, volume):
        return self.forward(volume)
