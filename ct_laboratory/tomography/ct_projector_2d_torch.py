# File: ct_projector_2d_torch.py
import torch

def compute_intersections_2d_torch(
    n_row: int,
    n_col: int,
    M: torch.Tensor,      # [2,2]
    b: torch.Tensor,      # [2]
    src: torch.Tensor,    # [n_ray, 2]
    dst: torch.Tensor,    # [n_ray, 2],
) -> torch.Tensor:
    """
    Returns sorted intersection parameters t in [0,1] for each ray, relative distance from source to detector
    crossing row=0..n_row or col=0..n_col lines.
    Output: [n_ray, n_intersections], where n_intersections = (n_row+1 + n_col+1).
    """

    # print('`hello')
    # import pdb; pdb.set_trace()

    device = src.device
    dtype  = src.dtype

    n_ray = src.shape[0]
    n_intersections = (n_row + 1) + (n_col + 1)

    # 1) Transform (x,y)->(row,col) => s_rc, d_rc
    M_inv = torch.inverse(M)
    M_inv_t = M_inv.transpose(0, 1)
    s_rc = (src - b) @ M_inv_t  # [n_ray,2]
    d_rc = (dst - b) @ M_inv_t
    dir_rc = d_rc - s_rc

    t_vals = torch.empty(n_ray, n_intersections, device=device, dtype=dtype)

    # Horizontal lines => row=0..n_row
    row_coords = torch.arange(n_row + 1, device=device, dtype=dtype).unsqueeze(0) - 0.5  # [1, n_row+1]
    row_s   = s_rc[:, 0].unsqueeze(1)  # [n_ray,1]
    row_dir = dir_rc[:, 0].unsqueeze(1)
    numerator_row   = row_coords - row_s
    denominator_row = row_dir
    zero_mask_row = (denominator_row == 0)
    t_row_all = numerator_row / denominator_row
    t_row_all = torch.where(zero_mask_row, torch.full_like(t_row_all, float('inf')), t_row_all)
    t_vals[:, : (n_row + 1)] = t_row_all

    # Vertical lines => col=0..n_col
    col_coords = torch.arange(n_col + 1, device=device, dtype=dtype).unsqueeze(0) - 0.5
    col_s   = s_rc[:, 1].unsqueeze(1)
    col_dir = dir_rc[:, 1].unsqueeze(1)
    numerator_col   = col_coords - col_s
    denominator_col = col_dir
    zero_mask_col = (denominator_col == 0)
    t_col_all = numerator_col / denominator_col
    t_col_all = torch.where(zero_mask_col, torch.full_like(t_col_all, float('inf')), t_col_all)
    t_vals[:, (n_row + 1):] = t_col_all

    # Keep only t in [0,1], else +inf
    out_of_range_mask = (t_vals < 0) | (t_vals > 1)
    t_vals = torch.where(out_of_range_mask, torch.full_like(t_vals, float('inf')), t_vals)

    # Sort each ray
    t_sorted, _ = torch.sort(t_vals, dim=1)

    return t_sorted


def forward_project_2d_torch(
    image: torch.Tensor,  # [n_row, n_col]
    t_sorted: torch.Tensor,  # [n_ray, n_intersections]
    M: torch.Tensor,      # [2,2], (row,col)->(x,y)
    b: torch.Tensor,      # [2],   offset
    src: torch.Tensor,    # [n_ray, 2]
    dst: torch.Tensor,    # [n_ray, 2]
) -> torch.Tensor:
    """
    Vectorized line-integral forward projector using:
      1) compute_intersections_t_sorted
      2) segment-by-segment accumulation with no explicit Python loops over rays.
    """
    
    # 1) get t-sorted intersections
    n_row, n_col = image.shape
    M_inv = torch.inverse(M)
    # t_sorted = compute_intersections_t_sorted(n_row, n_col, M_inv, b, src, dst)
    # shape => [n_ray, n_intersections], n_intersections = n_row+1 + n_col+1
    n_ray, n_intersections = t_sorted.shape

    # 2) Convert each sorted t => (x,y), shape => [n_ray, n_intersections, 2]
    s_expanded   = src.unsqueeze(1)          # [n_ray,1,2]
    dir_expanded = (dst - src).unsqueeze(1)  # [n_ray,1,2]
    t_expanded   = t_sorted.unsqueeze(-1)    # [n_ray,n_intersections,1]
    pts_xy = s_expanded + t_expanded * dir_expanded  # [n_ray,n_intersections,2]

    # 3) For each ray, consecutive pairs => segments
    #    We'll do:
    #        x0 = pts_xy[:, :-1, 0]
    #        x1 = pts_xy[:,  1:, 0]
    #    etc.
    #    shape => [n_ray, n_intersections-1]
    x0 = pts_xy[:, :-1, 0]
    x1 = pts_xy[:,  1:, 0]
    y0 = pts_xy[:, :-1, 1]
    y1 = pts_xy[:,  1:, 1]

    # 4) Identify valid segments => if x0 or x1 is inf => invalid
    #    We'll set them to zero in the final sum
    inf_mask = torch.isinf(x0) | torch.isinf(x1) | torch.isinf(y0) | torch.isinf(y1)
    # shape => [n_ray, n_intersections-1] (bool)

    # 5) segment length
    seg_len = torch.sqrt((x1 - x0)**2 + (y1 - y0)**2)  # [n_ray, n_intersections-1]

    # 6) midpoint
    mx = 0.5 * (x0 + x1)  # [n_ray, n_intersections-1]
    my = 0.5 * (y0 + y1)

    # 7) Convert midpoint to row,col
    #    rowf = M_inv[0,0]*(mx-b[0]) + M_inv[0,1]*(my-b[1])
    #    colf = M_inv[1,0]*(mx-b[0]) + M_inv[1,1]*(my-b[1])
    mx_shift = mx - b[0]
    my_shift = my - b[1]

    rowf = M_inv[0,0]*mx_shift + M_inv[0,1]*my_shift  # [n_ray, n_intersections-1]
    colf = M_inv[1,0]*mx_shift + M_inv[1,1]*my_shift

    # 8) Round row,col to nearest pixel. We'll do clamp in integer space so no out-of-bounds
    row_idx = torch.round(rowf).long()
    col_idx = torch.round(colf).long()

    # 9) Out-of-bounds mask
    out_of_bounds = (row_idx < 0) | (row_idx >= n_row) | (col_idx < 0) | (col_idx >= n_col)
    # shape => [n_ray, n_intersections-1]
    # final_mask => valid if not inf and not out_of_bounds
    valid_mask = (~inf_mask) & (~out_of_bounds)

    # 10) Gather pixel values
    # Flatten image: shape => [n_row * n_col]
    image_flat = image.view(-1)

    # Flatten row_idx,col_idx => map them to 1D
    # => index = row_idx*n_col + col_idx
    flat_idx = row_idx * n_col + col_idx  # shape => [n_ray, n_intersections-1]

    # We must ensure that invalid entries in `flat_idx` won't cause indexing errors.
    # We'll clamp them to 0 or do a where to set them to 0. Then multiply result by valid_mask afterward.
    flat_idx_clamped = torch.where(valid_mask, flat_idx, torch.zeros_like(flat_idx))

    # Now gather pixel values
    pixel_vals = image_flat[flat_idx_clamped]  # shape => [n_ray, n_intersections-1]
    # set invalid to 0
    pixel_vals = torch.where(valid_mask, pixel_vals, torch.zeros_like(pixel_vals))

    # 11) compute segment_val = pixel_val * seg_len
    segment_val = pixel_vals * seg_len

    # Also zero out any invalid segment
    segment_val = torch.where(valid_mask, segment_val, torch.zeros_like(segment_val))

    # 12) sum along the intersection dimension => line_integral
    line_integrals = segment_val.sum(dim=1)

    return line_integrals


def back_project_2d_torch(
    sinogram: torch.Tensor,   # [n_ray], line integral data
    t_sorted: torch.Tensor,   # [n_ray, n_intersections]
    M: torch.Tensor,          # [2,2]
    b: torch.Tensor,          # [2]
    src: torch.Tensor,        # [n_ray,2]
    dst: torch.Tensor,        # [n_ray,2]
    n_row: int,
    n_col: int
) -> torch.Tensor:
    """
    Vectorized back projector.
    Each ray r distributes sinogram[r]*segment_length among the voxels it crosses.
    Concurrency is handled by scatter_add_ so that multiple rays summing into the
    same voxel are combined properly.
    """

    device = sinogram.device
    dtype  = sinogram.dtype

    # 1) Get intersection parameters t in [0,1]
    M_inv = torch.inverse(M)
    # t_sorted = compute_intersections_t_sorted(n_row, n_col, M_inv, b, src, dst)
    n_ray, n_int = t_sorted.shape

    # 2) Intersection points => shape [n_ray, n_int, 2]
    s_expanded   = src.unsqueeze(1)           # [n_ray,1,2]
    dir_expanded = (dst - src).unsqueeze(1)   # [n_ray,1,2]
    t_expanded   = t_sorted.unsqueeze(-1)     # [n_ray,n_int,1]
    pts_xy       = s_expanded + t_expanded*dir_expanded

    # 3) Segment endpoints
    x0 = pts_xy[:, :-1, 0]  # [n_ray, n_int-1]
    x1 = pts_xy[:,  1:, 0]
    y0 = pts_xy[:, :-1, 1]
    y1 = pts_xy[:,  1:, 1]

    # Any intersection = inf => invalid
    inf_mask = torch.isinf(x0) | torch.isinf(x1) | torch.isinf(y0) | torch.isinf(y1)

    # 4) segment length
    seg_len = torch.sqrt((x1 - x0)**2 + (y1 - y0)**2)  # [n_ray, n_int-1]

    # 5) midpoint => for picking the voxel
    mx = 0.5*(x0 + x1)
    my = 0.5*(y0 + y1)

    # Convert midpoint => (row,col)
    mx_shift = mx - b[0]
    my_shift = my - b[1]
    rowf = M_inv[0,0]*mx_shift + M_inv[0,1]*my_shift
    colf = M_inv[1,0]*mx_shift + M_inv[1,1]*my_shift

    row_idx = torch.round(rowf).long()
    col_idx = torch.round(colf).long()

    # Out-of-bounds?
    oob_mask = (row_idx < 0)|(row_idx >= n_row)|(col_idx < 0)|(col_idx >= n_col)
    valid_mask = (~inf_mask) & (~oob_mask)

    # 6) The contribution from each segment is => sinogram[r]* seg_len
    # Expand sinogram => shape [n_ray,1], broadcast to [n_ray, n_int-1]
    sinogram_expanded = sinogram.view(n_ray, 1)
    contrib = sinogram_expanded * seg_len  # [n_ray, n_int-1]

    # Zero out invalid segments
    contrib = torch.where(valid_mask, contrib, torch.zeros_like(contrib))

    # 7) flatten row_idx,col_idx => 1D
    flat_idx = row_idx * n_col + col_idx  # shape => [n_ray, n_int-1]
    # clamp invalid => 0 to avoid out-of-bounds indexing
    flat_idx_clamped = torch.where(valid_mask, flat_idx, torch.zeros_like(flat_idx))

    # flatten everything => shape [n_ray*(n_int-1)]
    idx_1d = flat_idx_clamped.view(-1)
    val_1d = contrib.view(-1)

    # 8) scatter_add into a flat image
    out_flat = torch.zeros(n_row*n_col, device=device, dtype=dtype)
    # concurrency-safety => multiple threads can sum into the same index
    out_flat.scatter_add_(0, idx_1d, val_1d)

    # reshape => [n_row, n_col]
    out_image = out_flat.view(n_row, n_col)
    return out_image




class CTProjector2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image, M, b, src, dst):
        """
        ctx: PyTorch context for saving objects needed in backward pass.
        image: [n_row, n_col]
        M,b : geometry transform
        src,dst: [n_ray,2]
        Returns sinogram: [n_ray]
        """
        # Save geometry + shape for backward
        ctx.save_for_backward(M, b, src, dst)
        ctx.image_shape = image.shape  # (n_row, n_col)

        # forward project
        sinogram = forward_project_2d_torch(image, M, b, src, dst)
        return sinogram

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [n_ray], i.e. d(L)/d(sinogram)

        We want d(L)/d(image). We'll do a back projection of grad_output.
        The rest (M,b,src,dst) no gradient for now => None
        """
        M, b, src, dst = ctx.saved_tensors
        n_row, n_col = ctx.image_shape

        # back-project grad_output => shape [n_row, n_col]
        grad_image = back_project_2d_torch(
            grad_output, M, b, src, dst, n_row, n_col
        )
        # No grads for geometry
        return grad_image, None, None, None, None
    


class CTProjector2D(torch.nn.Module):
    def __init__(self, M, b, src, dst):
        """
        A,b: 2D transform
        src,dst: [n_ray,2]
        """
        super().__init__()
        self.register_buffer('M', M)
        self.register_buffer('b', b)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)

    def forward(self, image):
        """
        image: [n_row, n_col]
        returns sinogram [n_ray]
        with a grad that back-projects.
        """
        return CTProjector2DFunction.apply(image, self.M, self.b, self.src, self.dst)
