import torch

# Intersection computations
from .ct_projector_3d_torch import compute_intersections_3d_torch, compress_tvals_to_uint16
from .ct_projector_3d_cuda import compute_intersections_3d_cuda

# For Torch-based forward/back:
from .ct_projector_3d_torch import (
    forward_project_3d_torch,
    back_project_3d_torch
)

# For CUDA-based forward/back:
from .ct_projector_3d_cuda import (
    forward_project_3d_cuda,
    forward_project_3d_compressed_cuda,
    back_project_3d_cuda,
    back_project_3d_compressed_cuda
)

# Autograd function
from .ct_projector_3d_autograd import CTProjector3DFunction

# garbage collection
import gc


class CTProjector3DModule(torch.nn.Module):
    """
    CT projector module supporting precomputed or on-the-fly Siddon projection.
    """
    def __init__(self, n_x, n_y, n_z, M, b, src, dst,
                 backend='cuda', device=None, precomputed_intersections=False,
                 tvals=None, use_compression=True):
        """
        n_x, n_y, n_z: volume shape
        M, b: 3D transform (3x3, 3x1)
        src, dst: [n_ray, 3]
        backend: 'torch' or 'cuda'
        precomputed_intersections: if True, uses precomputed Siddon t-values
        tvals: optional precomputed intersection t-values. Can be float tensor or (uint16, start, scale) tuple.
        use_compression: if True, compresses precomputed tvals to uint16 to save memory.
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.backend = backend
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.use_compression = use_compression

        # Register geometry buffers
        self.register_buffer('M', M)
        self.register_buffer('b', b)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)
        
        # Initialize tvals buffers
        self.tvals_uint16 = None
        self.tvals_start = None
        self.tvals_scale = None
        self.tvals_full = None

        if tvals is not None:
            self.precomputed_intersections = True
            if isinstance(tvals, (tuple, list)):
                # Already compressed
                tu16, tstart, tscale = tvals
                self.tvals_uint16 = tu16
                self.tvals_start = tstart
                self.tvals_scale = tscale
            elif use_compression:
                # compress
                tu16, tstart, tscale = compress_tvals_to_uint16(tvals, self.M, self.src, self.dst)
                self.tvals_uint16 = tu16
                self.tvals_start = tstart
                self.tvals_scale = tscale
            else:
                self.tvals_full = tvals
                
        elif precomputed_intersections:
            self.precomputed_intersections = True
            # Compute on CPU to avoid GPU OOM for large projects
            if backend == 'torch' or True: # Force CPU for precomputation
                tvals_full = compute_intersections_3d_torch(n_x, n_y, n_z, M.cpu(), b.cpu(), src.cpu(), dst.cpu())
            else:
                tvals_full = compute_intersections_3d_cuda(n_x, n_y, n_z, M, b, src, dst)

            # Trim trailing INFINITY-only columns
            n_intersections = tvals_full.shape[1]
            for i in range(n_intersections):
                if torch.all(torch.isinf(tvals_full[:, i])):
                    tvals_full = tvals_full.narrow(1, 0, i)
                    break
            tvals_full = tvals_full.contiguous()
            
            if use_compression:
                # Compress on CPU
                tu16, tstart, tscale = compress_tvals_to_uint16(tvals_full, M.cpu(), src.cpu(), dst.cpu())
                    
                self.tvals_uint16 = tu16
                self.tvals_start = tstart
                self.tvals_scale = tscale
                del tvals_full
            else:
                self.tvals_full = tvals_full.to(device)
        
        # Ensure buffers are on device
        if device is not None:
            self.to(device)
        
        # Ensure buffers are on device
        if device is not None:
            self.to(device)

    def to(self, *args, **kwargs):
        """Override to() to handle tvals differently if needed."""
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        super().to(*args, **kwargs)
        
        # Compressed buffers
        if self.tvals_uint16 is not None:
            t_size = self.tvals_uint16.numel() * self.tvals_uint16.element_size()
            should_move = (self.backend == 'cuda') or (t_size < 2 * 1024**3)
            if should_move:
                self.tvals_uint16 = self.tvals_uint16.to(device, non_blocking=non_blocking)
                self.tvals_start = self.tvals_start.to(device, non_blocking=non_blocking)
                self.tvals_scale = self.tvals_scale.to(device, non_blocking=non_blocking)
        
        # Full buffer
        if self.tvals_full is not None:
            t_size = self.tvals_full.numel() * self.tvals_full.element_size()
            should_move = (self.backend == 'cuda') or (t_size < 2 * 1024**3)
            if should_move:
                self.tvals_full = self.tvals_full.to(device, non_blocking=non_blocking)
        
        return self

    def forward_project(self, volume):
        return self.forward(volume)
    
    def back_project(self, sinogram):
        t16 = self.tvals_uint16
        ts = self.tvals_start
        tsc = self.tvals_scale
        tf = self.tvals_full
        
        if self.backend == 'cuda':
             device = sinogram.device
             if t16 is not None and not t16.is_cuda:
                 self.tvals_uint16, self.tvals_start, self.tvals_scale = t16.to(device), ts.to(device), tsc.to(device)
             if tf is not None and not tf.is_cuda:
                 self.tvals_full = tf.to(device)
                 
        t16, ts, tsc, tf = self.tvals_uint16, self.tvals_start, self.tvals_scale, self.tvals_full

        if self.backend == 'torch':
            t_arg = (t16, ts, tsc) if t16 is not None else tf
            volume = back_project_3d_torch(sinogram, t_arg, self.M, self.b, self.src, self.dst, self.n_x, self.n_y, self.n_z)
        else:
            from .ct_projector_3d_cuda import back_project_3d_compressed_cuda, back_project_3d_cuda
            if t16 is not None:
                volume = back_project_3d_compressed_cuda(sinogram, t16, ts, tsc, self.M, self.b, self.src, self.dst, self.n_x, self.n_y, self.n_z)
            else:
                volume = back_project_3d_cuda(sinogram, tf, self.src, self.dst, self.M, self.b, self.n_x, self.n_y, self.n_z)
        return volume

    def forward(self, volume):
        t16 = self.tvals_uint16
        ts = self.tvals_start
        tsc = self.tvals_scale
        tf = self.tvals_full
        
        if self.backend == 'cuda':
            device = volume.device
            if t16 is not None and not t16.is_cuda:
                self.tvals_uint16, self.tvals_start, self.tvals_scale = t16.to(device), ts.to(device), tsc.to(device)
            if tf is not None and not tf.is_cuda:
                self.tvals_full = tf.to(device)

        t16, ts, tsc, tf = self.tvals_uint16, self.tvals_start, self.tvals_scale, self.tvals_full

        return CTProjector3DFunction.apply(
            volume, t16 if t16 is not None else tf, ts, tsc, self.M, self.b, self.src, self.dst, self.backend
        )

    def save_tvals(self, filePath):
        """Save precomputed t-values to a file."""
        if self.tvals_uint16 is None and self.tvals_full is None:
            raise ValueError("No precomputed t-values to save.")
        
        tvals_dict = {
            'tvals_uint16': self.tvals_uint16.cpu() if self.tvals_uint16 is not None else None,
            'tvals_start': self.tvals_start.cpu() if self.tvals_start is not None else None,
            'tvals_scale': self.tvals_scale.cpu() if self.tvals_scale is not None else None,
            'tvals_full': self.tvals_full.cpu() if self.tvals_full is not None else None,
            'n_x': self.n_x, 'n_y': self.n_y, 'n_z': self.n_z,
            'n_ray': self.src.shape[0]
        }
        torch.save(tvals_dict, filePath)

    def load_tvals(self, filePath, device=None):
        if device is None: device = self.src.device
        d = torch.load(filePath, map_location='cpu')
        
        if d['n_x'] != self.n_x or d['n_y'] != self.n_y or d['n_z'] != self.n_z:
            raise ValueError("Dimension mismatch")
            
        self.register_buffer('tvals_uint16', d.get('tvals_uint16').to(device) if d.get('tvals_uint16') is not None else None)
        self.register_buffer('tvals_start', d.get('tvals_start').to(device) if d.get('tvals_start') is not None else None)
        self.register_buffer('tvals_scale', d.get('tvals_scale').to(device) if d.get('tvals_scale') is not None else None)
        self.register_buffer('tvals_full', d.get('tvals_full').to(device) if d.get('tvals_full') is not None else None)
        self.precomputed_intersections = True

    def set_tvals(self, tvals, device=None):
        if device is None: device = self.src.device
        if tvals is None:
            self.tvals_uint16 = self.tvals_start = self.tvals_scale = self.tvals_full = None
            self.precomputed_intersections = False
            return

        if isinstance(tvals, (tuple, list)):
            tu16, tstart, tscale = tvals
            self.register_buffer('tvals_uint16', tu16.to(device))
            self.register_buffer('tvals_start', tstart.to(device))
            self.register_buffer('tvals_scale', tscale.to(device))
            self.tvals_full = None
        else:
            if self.use_compression:
                tu16, tstart, tscale = compress_tvals_to_uint16(tvals, self.M, self.src, self.dst)
                self.register_buffer('tvals_uint16', tu16.to(device))
                self.register_buffer('tvals_start', tstart.to(device))
                self.register_buffer('tvals_scale', tscale.to(device))
                self.tvals_full = None
            else:
                self.register_buffer('tvals_full', tvals.to(device))
                self.tvals_uint16 = self.tvals_start = self.tvals_scale = None
        
        self.precomputed_intersections = True

    @property
    def tvals(self):
        if self.tvals_uint16 is not None:
            from .ct_projector_3d_torch import decompress_tvals_from_uint16
            return decompress_tvals_from_uint16(self.tvals_uint16, self.tvals_start, self.tvals_scale)
        return self.tvals_full

    @tvals.setter
    def tvals(self, value):
        self.set_tvals(value)


def precompute_tvals_stitched(
    n_x, n_y, n_z, M, b, src_all, dst_all, 
    chunk_size=1000000, backend='cuda', device=None, verbose=True, use_compression=True
):
    """
    Compute tvals in smaller chunks to avoid GPU OOM during precomputation.
    """
    import math
    if device is None:
        device = torch.device("cuda" if backend == 'cuda' and torch.cuda.is_available() else "cpu")
    
    n_ray = src_all.shape[0]
    n_max_int = (n_x + 1) + (n_y + 1) + (n_z + 1)
    
    if verbose:
        print(f"Precomputing {n_ray} rays in chunks of {chunk_size} using {backend}...")

    # Pre-allocate
    if use_compression:
        main_cpu = torch.zeros((n_ray, n_max_int), dtype=torch.int16)
        starts_cpu = torch.zeros(n_ray, dtype=torch.float32)
        scales_cpu = torch.zeros(n_ray, dtype=torch.float32)
    else:
        full_cpu = torch.zeros((n_ray, n_max_int), dtype=torch.float32)

    n_chunks = math.ceil(n_ray / chunk_size)
    from .ct_projector_3d_torch import compute_intersections_3d_torch, compress_tvals_to_uint16
    from .ct_projector_3d_cuda import compute_intersections_3d_cuda

    max_cols = 0
    for i in range(n_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, n_ray)
        src_c, dst_c = src_all[start:end].to(device), dst_all[start:end].to(device)
        M_d, b_d = M.to(device), b.to(device)
        
        t_c = compute_intersections_3d_torch(n_x, n_y, n_z, M_d, b_d, src_c, dst_c) if backend == 'torch' else \
              compute_intersections_3d_cuda(n_x, n_y, n_z, M_d, b_d, src_c, dst_c)
            
        max_cols = max(max_cols, t_c.shape[1])
        
        if use_compression:
            if t_c.is_cuda:
                from .ct_projector_3d_cuda import compress_tvals_3d_cuda
                t16, ts, tsc = compress_tvals_3d_cuda(t_c, M_d, src_c, dst_c)
            else:
                t16, ts, tsc = compress_tvals_to_uint16(t_c, M_d, src_c, dst_c)
            main_cpu[start:end, :t16.shape[1]] = t16.cpu()
            starts_cpu[start:end] = ts.cpu()
            scales_cpu[start:end] = tsc.cpu()
        else:
            full_cpu[start:end, :t_c.shape[1]] = t_c.cpu()
        
        del src_c, dst_c, t_c
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    if use_compression:
        return main_cpu[:, :max_cols].contiguous(), starts_cpu, scales_cpu
    else:
        return full_cpu[:, :max_cols].contiguous()


