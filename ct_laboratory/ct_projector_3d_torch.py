# File: ct_projector_3d_torch.py
import torch

def compress_tvals_to_uint16(tvals: torch.Tensor, M: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    """
    Compress sorted tvals [N, K] into uint16 deltas based on voxel geometry.
    This provides significantly higher precision than uint8 and avoids clamping issues for moderate gaps.
    tvals are in [0, 1].
    Args:
        tvals: [N, K] sorted intersection params
        M: [3, 3] voxel to world transform. Used to determine max voxel size.
        src, dst: [N, 3] World coordinates.
    Returns:
        tvals_uint16: [N, K] (first col is 0, rest are diffs). dtype=torch.int16 (used as uint16)
        starts: [N] (float32)
        scales: [N] (float32)
    """
    if tvals.numel() == 0:
        return tvals.to(torch.int16), torch.empty(0, device=tvals.device), torch.empty(0, device=tvals.device)

    # 1. Determine Scale Factor (Resolution)
    # We want to map the voxel diagonal to a safe range within uint16.
    # We choose 10000 to represent the voxel diagonal.
    # This leaves headroom before clipping occurs.
    TARGET_VAL = 10000.0
    
    # 4 diagonals
    diags = torch.tensor([
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1]
    ], device=M.device, dtype=M.dtype).t()
    voxel_diags = M @ diags
    voxel_diag_len = torch.norm(voxel_diags, dim=0).max()
    
    # Ray lengths
    ray_lens = torch.norm(dst - src, dim=1)
    
    # Scale factor 
    # 1 unit = (voxel_diag_len / TARGET_VAL) in world space
    ray_lens = ray_lens.clamp(min=1e-6)
    scales = (voxel_diag_len / TARGET_VAL) / ray_lens
    
    N, K = tvals.shape
    # Handle INF values by substituting a large finite value (10.0) which will be OOB
    tvals_finite = torch.where(torch.isinf(tvals), torch.tensor(10.0, device=tvals.device, dtype=tvals.dtype), tvals)
    
    starts = tvals_finite[:, 0]
    tvals_uint16 = torch.zeros((N, K), device=tvals.device, dtype=torch.int16)
    k_sum_prev = torch.zeros(N, device=tvals.device, dtype=torch.float32)
    inv_scales = 1.0 / scales
    
    for i in range(1, K):
        k_sum_curr = torch.round((tvals_finite[:, i] - starts) * inv_scales)
        
        # Clamp to 32000 to stay safely within int16 positive range if needed,
        # but we'll use 65535 if we were truly unsigned. Torch int16 is signed.
        # Let's stay with 32000 for safety with signed int16.
        diff = (k_sum_curr - k_sum_prev).clamp(0, 32000).to(torch.int16)
        
        tvals_uint16[:, i] = diff
        k_sum_prev += diff.float()
    
    return tvals_uint16, starts, scales


def decompress_tvals_from_uint16(tvals_uint16, starts, scales):
    """
    Decompress uint16 tvals [N, K].
    """
    t_recon = tvals_uint16.float().cumsum_(dim=1)
    t_recon.mul_(scales.unsqueeze(1))
    t_recon.add_(starts.unsqueeze(1))
    return t_recon


def compute_intersections_3d_torch(
    n_x: int,
    n_y: int,
    n_z: int,
    M: torch.Tensor,      # [3,3]
    b: torch.Tensor,      # [3]
    src: torch.Tensor,    # [n_rMy, 3]
    dst: torch.Tensor,    # [n_ray, 3]
) -> torch.Tensor:
    """
    Returns sorted intersection parameters t in [0,1] for each 3D ray.
    The grid has shape [n_x, n_y, n_z].
    Planes are at x=i-0.5, i=0..n_x;  y=j-0.5, j=0..n_y;  z=k-0.5, k=0..n_z.
    Output shape: [n_ray, n_intersections], where n_intersections = (n_x+1 + n_y+1 + n_z+1).
    """

    device = src.device
    dtype  = src.dtype
    n_ray  = src.shape[0]

    # 1) Transform src/dst from (x,y,z) to (i,j,k)
    #    i = M_inv*(xyz - b). We'll do it by (src - b) @ M_inv^T
    M_inv = torch.inverse(M)
    M_inv_t = M_inv.transpose(0, 1)  # so we can do (vector) @ (M_inv^T)
    s_ijk = (src - b) @ M_inv_t  # [n_ray,3]
    d_ijk = (dst - b) @ M_inv_t
    dir_ijk = d_ijk - s_ijk  # [n_ray,3]

    # 2) We'll build up a [n_ray, n_x+1 + n_y+1 + n_z+1] array of t-values
    n_intersections = (n_x+1) + (n_y+1) + (n_z+1)
    t_vals = torch.empty(n_ray, n_intersections, device=device, dtype=dtype)

    # For convenience, compute each set of planes separately, then fill t_vals
    # We'll do: 
    #   x-planes at i-0.5 => i=0..n_x
    #   y-planes at j-0.5 => j=0..n_y
    #   z-planes at k-0.5 => k=0..n_z

    # --- X-planes ---
    x_coords = torch.arange(n_x+1, device=device, dtype=dtype).unsqueeze(0) - 0.5
    # shape => [1, n_x+1]
    s_x = s_ijk[:, 0].unsqueeze(1)    # [n_ray,1]
    dir_x = dir_ijk[:, 0].unsqueeze(1)
    numerator_x = x_coords - s_x      # [n_ray, n_x+1]
    denominator_x = dir_x
    zero_mask_x = (denominator_x.abs() < 1.0e-12)
    t_x_all = numerator_x / denominator_x
    t_x_all = torch.where(zero_mask_x, torch.full_like(t_x_all, float('inf')), t_x_all)

    # --- Y-planes ---
    y_coords = torch.arange(n_y+1, device=device, dtype=dtype).unsqueeze(0) - 0.5
    s_y = s_ijk[:, 1].unsqueeze(1)
    dir_y = dir_ijk[:, 1].unsqueeze(1)
    numerator_y = y_coords - s_y
    denominator_y = dir_y
    zero_mask_y = (denominator_y.abs() < 1.0e-12)
    t_y_all = numerator_y / denominator_y
    t_y_all = torch.where(zero_mask_y, torch.full_like(t_y_all, float('inf')), t_y_all)

    # --- Z-planes ---
    z_coords = torch.arange(n_z+1, device=device, dtype=dtype).unsqueeze(0) - 0.5
    s_z = s_ijk[:, 2].unsqueeze(1)
    dir_z = dir_ijk[:, 2].unsqueeze(1)
    numerator_z = z_coords - s_z
    denominator_z = dir_z
    zero_mask_z = (denominator_z.abs() < 1.0e-12)
    t_z_all = numerator_z / denominator_z
    t_z_all = torch.where(zero_mask_z, torch.full_like(t_z_all, float('inf')), t_z_all)

    # Put them all into t_vals
    t_vals[:, : (n_x+1)] = t_x_all
    t_vals[:, (n_x+1) : (n_x+1 + n_y+1)] = t_y_all
    t_vals[:, (n_x+1 + n_y+1) : ] = t_z_all

    # 3) Filter out t<0 or t>1 => set to +inf
    out_of_range_mask = (t_vals < 0) | (t_vals > 1)
    t_vals = torch.where(out_of_range_mask, torch.full_like(t_vals, float('inf')), t_vals)

    # 4) Sort
    t_sorted, _ = torch.sort(t_vals, dim=1)
    return t_sorted


def forward_project_3d_torch(
    volume: torch.Tensor,  # [nx, ny, nz]
    t_sorted_arg,          # Tensor or tuple(t_uint16, t_start, t_scale)
    M: torch.Tensor,       # [3,3]
    b: torch.Tensor,       # [3]
    src: torch.Tensor,     # [n_ray, 3]
    dst: torch.Tensor,     # [n_ray, 3]
) -> torch.Tensor:
    """
    3D forward projector using purely PyTorch ops (no custom CUDA kernel).
    Updates to support chunking and compressed tvals to save memory.
    """
    import math

    device = volume.device
    dtype  = volume.dtype
    n_x, n_y, n_z = volume.shape
    M_inv = torch.inverse(M)

    # Check for compressed tvals
    if isinstance(t_sorted_arg, (tuple, list)):
        t_uint16, t_start, t_scale = t_sorted_arg
        n_ray = t_uint16.shape[0]
        compressed = True
    else:
        t_sorted = t_sorted_arg
        n_ray = t_sorted.shape[0]
        compressed = False

    sinogram_all = torch.zeros(n_ray, device=device, dtype=dtype)
    
    # Process in larger chunks for GPU efficiency
    chunk_size = 65536 
    n_chunks = math.ceil(n_ray / chunk_size)

    # Pre-fetch volume flat for indexing access
    volume_flat = volume.view(-1)
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(n_ray, (i + 1) * chunk_size)
        
        # 1. Get t-values for this chunk
        if compressed:
            # Move compressed chunks to the same device as the volume for processing
            t_u16_batch = t_uint16[start:end].to(device, non_blocking=True)
            t_s_batch = t_start[start:end].to(device, non_blocking=True)
            t_sc_batch = t_scale[start:end].to(device, non_blocking=True)
            
            t_chunk = decompress_tvals_from_uint16(
                t_u16_batch, t_s_batch, t_sc_batch
            )
        else:
            t_chunk = t_sorted[start:end].to(device, non_blocking=True)
            
        src_chunk = src[start:end].to(device, non_blocking=True)
        dst_chunk = dst[start:end].to(device, non_blocking=True)
        
        # 2. Points along rays
        s_expanded = src_chunk.unsqueeze(1)
        dir_expanded = (dst_chunk - src_chunk).unsqueeze(1)
        t_expanded = t_chunk.unsqueeze(-1)
        pts_xyz = s_expanded + t_expanded * dir_expanded

        # 3. Segments
        x0 = pts_xyz[:, :-1, 0]
        x1 = pts_xyz[:,  1:, 0]
        y0 = pts_xyz[:, :-1, 1]
        y1 = pts_xyz[:,  1:, 1]
        z0 = pts_xyz[:, :-1, 2]
        z1 = pts_xyz[:,  1:, 2]

        inf_mask = torch.isinf(x0) | torch.isinf(x1) \
                 | torch.isinf(y0) | torch.isinf(y1) \
                 | torch.isinf(z0) | torch.isinf(z1)

        seg_len = torch.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        mz = 0.5 * (z0 + z1)

        # 4. Indices
        mx_shift = mx - b[0]
        my_shift = my - b[1]
        mz_shift = mz - b[2]

        i_f = M_inv[0,0]*mx_shift + M_inv[0,1]*my_shift + M_inv[0,2]*mz_shift
        j_f = M_inv[1,0]*mx_shift + M_inv[1,1]*my_shift + M_inv[1,2]*mz_shift
        k_f = M_inv[2,0]*mx_shift + M_inv[2,1]*my_shift + M_inv[2,2]*mz_shift

        i_idx = torch.round(i_f).long()
        j_idx = torch.round(j_f).long()
        k_idx = torch.round(k_f).long()

        oob_mask = (i_idx < 0)|(i_idx >= n_x) \
                 | (j_idx < 0)|(j_idx >= n_y) \
                 | (k_idx < 0)|(k_idx >= n_z)

        valid_mask = (~inf_mask) & (~oob_mask)

        flat_idx = (i_idx * (n_y*n_z) + j_idx * n_z + k_idx)
        flat_idx_clamped = torch.where(valid_mask, flat_idx, torch.zeros_like(flat_idx))

        pixel_vals = volume_flat[flat_idx_clamped]
        pixel_vals = torch.where(valid_mask, pixel_vals, torch.zeros_like(pixel_vals))

        segment_val = pixel_vals * seg_len
        segment_val = torch.where(valid_mask, segment_val, torch.zeros_like(segment_val))

        line_integrals = segment_val.sum(dim=1)
        sinogram_all[start:end] = line_integrals
        
    return sinogram_all


def back_project_3d_torch(
    sinogram: torch.Tensor,  # [n_ray]
    t_sorted_arg,            # Tensor or tuple
    M: torch.Tensor,
    b: torch.Tensor,
    src: torch.Tensor,  # [n_ray,3]
    dst: torch.Tensor,  # [n_ray,3]
    n_x: int,
    n_y: int,
    n_z: int
) -> torch.Tensor:
    """
    3D back projector, purely in PyTorch.
    Updates to support chunking and compressed tvals.
    """
    import math
    
    device = sinogram.device
    dtype  = sinogram.dtype
    
    # Check for compressed tvals
    if isinstance(t_sorted_arg, (tuple, list)):
        t_uint16, t_start, t_scale = t_sorted_arg
        n_ray = t_uint16.shape[0]
        compressed = True
    else:
        t_sorted = t_sorted_arg
        n_ray = t_sorted.shape[0]
        compressed = False
        
    M_inv = torch.inverse(M)
    
    # We'll accumulate into a flat volume using scatter_add
    # Note: scatter_add on float32/cuda is atomic (mostly).
    out_flat = torch.zeros(n_x*n_y*n_z, dtype=dtype, device=device)
    
    chunk_size = 65536
    n_chunks = math.ceil(n_ray / chunk_size)
    
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(n_ray, (i + 1) * chunk_size)
        
        # 1. Get t-values for this chunk
        if compressed:
            # Move compressed chunks to the same device as sinogram for processing
            t_u16_batch = t_uint16[start:end].to(device, non_blocking=True)
            t_s_batch = t_start[start:end].to(device, non_blocking=True)
            t_sc_batch = t_scale[start:end].to(device, non_blocking=True)

            t_chunk = decompress_tvals_from_uint16(
                t_u16_batch, t_s_batch, t_sc_batch
            )
        else:
            t_chunk = t_sorted[start:end].to(device, non_blocking=True)
        
        src_chunk = src[start:end].to(device, non_blocking=True)
        dst_chunk = dst[start:end].to(device, non_blocking=True)
        sino_chunk = sinogram[start:end]

        # 2. Points
        s_expanded = src_chunk.unsqueeze(1)
        dir_expanded = (dst_chunk - src_chunk).unsqueeze(1)
        t_expanded = t_chunk.unsqueeze(-1)
        pts_xyz = s_expanded + t_expanded*dir_expanded

        # 3. Segments
        x0 = pts_xyz[:, :-1, 0]
        x1 = pts_xyz[:,  1:, 0]
        y0 = pts_xyz[:, :-1, 1]
        y1 = pts_xyz[:,  1:, 1]
        z0 = pts_xyz[:, :-1, 2]
        z1 = pts_xyz[:,  1:, 2]

        inf_mask = torch.isinf(x0)|torch.isinf(x1)|torch.isinf(y0)|torch.isinf(y1)|torch.isinf(z0)|torch.isinf(z1)
        seg_len = torch.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

        mx = 0.5*(x0 + x1)
        my = 0.5*(y0 + y1)
        mz = 0.5*(z0 + z1)

        mx_shift = mx - b[0]
        my_shift = my - b[1]
        mz_shift = mz - b[2]

        i_f = M_inv[0,0]*mx_shift + M_inv[0,1]*my_shift + M_inv[0,2]*mz_shift
        j_f = M_inv[1,0]*mx_shift + M_inv[1,1]*my_shift + M_inv[1,2]*mz_shift
        k_f = M_inv[2,0]*mx_shift + M_inv[2,1]*my_shift + M_inv[2,2]*mz_shift

        i_idx = torch.round(i_f).long()
        j_idx = torch.round(j_f).long()
        k_idx = torch.round(k_f).long()

        oob_mask = (i_idx<0)|(i_idx>=n_x)|(j_idx<0)|(j_idx>=n_y)|(k_idx<0)|(k_idx>=n_z)
        valid_mask = (~inf_mask) & (~oob_mask)

        # Contribution
        contrib = sino_chunk.view(-1, 1) * seg_len
        contrib = torch.where(valid_mask, contrib, torch.zeros_like(contrib))

        flat_idx = i_idx*(n_y*n_z) + j_idx*n_z + k_idx
        flat_idx_clamped = torch.where(valid_mask, flat_idx, torch.zeros_like(flat_idx))

        idx_1d = flat_idx_clamped.view(-1)
        val_1d = contrib.view(-1)

        out_flat.scatter_add_(0, idx_1d, val_1d)

    volume = out_flat.view(n_x, n_y, n_z)
    return volume


class CTProjector3DFunction(torch.autograd.Function):
    """
    A convenience autograd Function that does 3D forward/back in PyTorch.
    """
    @staticmethod
    def forward(ctx, volume, M, b, src, dst):
        ctx.save_for_backward(M, b, src, dst)
        ctx.volume_shape = volume.shape  # (n_x,n_y,n_z)
        sinogram = forward_project_3d_torch(volume, None, M, b, src, dst)
        return sinogram

    @staticmethod
    def backward(ctx, grad_output):
        M, b, src, dst = ctx.saved_tensors
        n_x, n_y, n_z = ctx.volume_shape
        grad_volume = back_project_3d_torch(
            grad_output, None, M, b, src, dst, n_x, n_y, n_z
        )
        return grad_volume, None, None, None, None


class CTProjector3D(torch.nn.Module):
    def __init__(self, M, b, src, dst):
        super().__init__()
        self.register_buffer('M', M)
        self.register_buffer('b', b)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)
    def forward(self, volume):
        return CTProjector3DFunction.apply(volume, self.M, self.b, self.src, self.dst)
