# File: ct_projector_3d_torch.py
import torch

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
    t_sorted: torch.Tensor,  # [n_ray, n_int]
    M: torch.Tensor,       # [3,3], (i,j,k)->(x,y,z)
    b: torch.Tensor,       # [3]
    src: torch.Tensor,     # [n_ray, 3]
    dst: torch.Tensor,     # [n_ray, 3]
) -> torch.Tensor:
    """
    3D forward projector using purely PyTorch ops (no custom CUDA kernel).
    volume: [n_x, n_y, n_z]
    M,b: transform from (i,j,k) -> (x,y,z)
    src,dst: [n_ray,3]
    Returns: [n_ray] line integrals
    """

    device = volume.device
    dtype  = volume.dtype

    n_x, n_y, n_z = volume.shape

    # 1) Compute sorted intersection params
    M_inv = torch.inverse(M)
    # t_sorted = compute_intersections_t_sorted_3d(n_x, n_y, n_z, M_inv, b, src, dst)
    n_ray, n_int = t_sorted.shape

    # 2) Evaluate segment midpoints and lengths
    s_expanded = src.unsqueeze(1)            # [n_ray,1,3]
    dir_expanded = (dst - src).unsqueeze(1)  # [n_ray,1,3]
    t_expanded = t_sorted.unsqueeze(-1)      # [n_ray, n_int, 1]
    pts_xyz = s_expanded + t_expanded * dir_expanded  # [n_ray, n_int, 3]

    # segment endpoints
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

    # midpoint
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    mz = 0.5 * (z0 + z1)

    # Convert midpoint => (i,j,k)
    M_inv = torch.inverse(M)  # shape [3,3]
    # shift by b
    mx_shift = mx - b[0]
    my_shift = my - b[1]
    mz_shift = mz - b[2]

    # i = M_inv[0,0]*mx_shift + M_inv[0,1]*my_shift + M_inv[0,2]*mz_shift
    # similarly j, k
    # do in one shot:
    #   [i,j,k] = [mx_shift,my_shift,mz_shift] @ M_inv^T
    # or we do each one manually:
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

    volume_flat = volume.view(-1)
    flat_idx = (i_idx * (n_y*n_z) + j_idx * n_z + k_idx)
    flat_idx_clamped = torch.where(valid_mask, flat_idx, torch.zeros_like(flat_idx))

    pixel_vals = volume_flat[flat_idx_clamped]
    pixel_vals = torch.where(valid_mask, pixel_vals, torch.zeros_like(pixel_vals))

    segment_val = pixel_vals * seg_len
    segment_val = torch.where(valid_mask, segment_val, torch.zeros_like(segment_val))

    # sum along segments
    line_integrals = segment_val.sum(dim=1)
    return line_integrals


def back_project_3d_torch(
    sinogram: torch.Tensor,  # [n_ray]
    t_sorted: torch.Tensor,  # [n_ray, n_int]
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
    sinogram: [n_ray] (no batch dimension in this simplified example).
    Returns: volume [n_x,n_y,n_z].
    """

    device = sinogram.device
    dtype  = sinogram.dtype
    n_ray  = sinogram.shape[0]

    # 1) intersection parameters
    M_inv = torch.inverse(M)
    # t_sorted = compute_intersections_t_sorted_3d(n_x, n_y, n_z, M_inv, b, src, dst)
    n_int = t_sorted.shape[1]

    s_expanded = src.unsqueeze(1)
    dir_expanded = (dst - src).unsqueeze(1)
    t_expanded = t_sorted.unsqueeze(-1)
    pts_xyz = s_expanded + t_expanded*dir_expanded

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

    # Each segment has a contribution => sinogram[r]*seg_len
    sinogram_expanded = sinogram.view(n_ray,1)
    contrib = sinogram_expanded * seg_len
    contrib = torch.where(valid_mask, contrib, torch.zeros_like(contrib))

    flat_idx = i_idx*(n_y*n_z) + j_idx*n_z + k_idx
    flat_idx_clamped = torch.where(valid_mask, flat_idx, torch.zeros_like(flat_idx))

    # We'll scatter_add into a flat volume
    out_flat = torch.zeros(n_x*n_y*n_z, dtype=dtype, device=device)
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
        sinogram = forward_project_3d_torch(volume, M, b, src, dst)
        return sinogram

    @staticmethod
    def backward(ctx, grad_output):
        M, b, src, dst = ctx.saved_tensors
        n_x, n_y, n_z = ctx.volume_shape
        grad_volume = back_project_3d_torch(
            grad_output, M, b, src, dst, n_x, n_y, n_z
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
