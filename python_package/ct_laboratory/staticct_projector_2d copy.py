import math
import torch
from typing import Optional

# Reuse the existing 2D projector base class that you have, e.g.:
# from .ct_projector_2d_module import CTProjector2DModule

def build_static_2d_geometry(
    source_positions: torch.Tensor,    # [n_source, 2], in gantry coords
    module_center_positions: torch.Tensor,      # [n_module, 2], in gantry coords
    module_orientations: torch.Tensor, # [n_module, 2, 2], local -> gantry
    det_n_col: int,
    det_spacing: float,
    source_module_mask: torch.Tensor,  # [n_source, n_module] bool
    M_gantry: torch.Tensor,            # [n_frame, 2, 2]
    b_gantry: torch.Tensor,            # [n_frame, 2]
    active_sources: torch.Tensor,      # [n_frame, n_source] bool
) -> (torch.Tensor, torch.Tensor):
    r"""
    Constructs a 2D array of (src, dst) rays from:
    1) A set of static sources, modules in *gantry coordinates*.
    2) A per-frame gantry->image transform: x_img = M_gantry[i] @ x_gantry + b_gantry[i].
    3) A boolean mask indicating which sources are active per frame.
    4) A boolean mask indicating which modules each source illuminates.

    Each module is assumed to have `det_n_col` pixels, spaced by `det_spacing`,
    oriented according to `module_orientations[m_idx]`, and centered at `module_centers[m_idx]`.

    Returns
    -------
    src_out : torch.Tensor, shape = [N, 2]
    dst_out : torch.Tensor, shape = [N, 2]
        Where `N` is the total number of active rays across all frames, sources, modules, and detector columns.
    """

    device = source_positions.device
    n_frame = M_gantry.shape[0]
    n_source = source_positions.shape[0]
    n_module = module_center_positions.shape[0]

    # Create local pixel coordinate offsets for each module (1D in this example).
    # For convenience, we'll call them columns in local space => shape: [n_col, 2].
    col_positions_local = []
    mid_col = (det_n_col - 1) / 2.0
    for ic in range(det_n_col):
        offset_c = (ic - mid_col) * det_spacing
        # local = (offset_c, 0)
        col_positions_local.append([offset_c, 0.0])
    col_positions_local = torch.tensor(col_positions_local, dtype=torch.float32, device=device)  # [n_col,2]

    col_positions_gantry = col_positions_local @ module_orientations.transpose(1, 2) + module_center_positions.unsqueeze(1)  # [n_module, n_col, 2]

    all_src = []
    all_dst = []

    for i_frame in range(n_frame):
        Mf = M_gantry[i_frame]  # [2,2]
        bf = b_gantry[i_frame]  # [2]

        active_src_mask = active_sources[i_frame]  # shape [n_source]

        # For each source
        for s_idx in range(n_source):
            if not bool(active_src_mask[s_idx]):
                continue
            # source in gantry coords => transform to image
            src_gantry = source_positions[s_idx]         # [2]
            src_image = Mf @ src_gantry + bf             # [2]

            # For each module that is illuminated by this source
            for m_idx in range(n_module):
                if not bool(source_module_mask[s_idx, m_idx]):
                    continue

                module_center_positions_gantry = module_center_positions[m_idx]          # [2]
                Rm = module_orientations[m_idx]                # [2,2]

                # local -> gantry => pixel_gantry = Rm @ local_col + center_gantry
                # then gantry -> image => pixel_image = Mf @ pixel_gantry + bf

                # Batched
                local_cols = col_positions_local  # shape [n_col, 2]
                pixel_gantry = local_cols @ Rm.transpose(0, 1) + module_center_positions_gantry  # [n_col,2]
                pixel_image = pixel_gantry @ Mf.transpose(0, 1) + bf            # [n_col,2]

                # Expand src_image to match pixel_image shape
                src_batch = src_image.unsqueeze(0).expand_as(pixel_image)  # [n_col,2]

                all_src.append(src_batch)
                all_dst.append(pixel_image)

    if len(all_src) == 0:
        # No active rays
        src_out = torch.empty((0, 2), dtype=torch.float32, device=device)
        dst_out = torch.empty((0, 2), dtype=torch.float32, device=device)
        return src_out, dst_out

    src_out = torch.cat(all_src, dim=0)  # [N,2]
    dst_out = torch.cat(all_dst, dim=0)  # [N,2]
    return src_out, dst_out


# -------------------------------------------------------------------
#   Main 2D static projector class
# -------------------------------------------------------------------
# Suppose you have a "CTProjector2DModule" which expects:
#   - n_row, n_col for the image
#   - M,b for (row,col)->(x,y)
#   - src, dst for each ray
#   - etc.
# This class will wrap that logic.
# -------------------------------------------------------------------
from .ct_projector_2d_module import CTProjector2DModule

class StaticCTProjector2D(CTProjector2DModule):
    r"""
    2D static gantry projector:
      - Accepts an image of shape (n_row, n_col)
      - We store a large (src, dst) set for all frames, then rely on the standard
        intersection-based 2D projector from CTProjector2DModule.
    """
    def __init__(
        self,
        n_row: int,
        n_col: int,
        M_gantry: torch.Tensor,       # [n_frame, 2, 2]
        b_gantry: torch.Tensor,       # [n_frame, 2]
        source_positions: torch.Tensor,       # [n_source, 2]
        module_centers: torch.Tensor,         # [n_module, 2]
        module_orientations: torch.Tensor,    # [n_module, 2,2]
        det_n_col: int,
        det_spacing: float,
        source_module_mask: torch.Tensor,     # [n_source, n_module] bool
        active_sources: torch.Tensor,         # [n_frame, n_source] bool
        pixel_size: float = 1.0,
        backend: str = "cuda",
        device: Optional[torch.device] = None
    ):
        # Build final (src,dst) for all frames
        src_all, dst_all = build_static_2d_geometry(
            source_positions=source_positions,
            module_center_positions=module_centers,
            module_orientations=module_orientations,
            det_n_col=det_n_col,
            det_spacing=det_spacing,
            source_module_mask=source_module_mask,
            M_gantry=M_gantry,
            b_gantry=b_gantry,
            active_sources=active_sources
        )

        # Next: define transform from (row,col)->(x,y).
        # e.g. center of the image at (0,0):
        row_mid = (n_row - 1) / 2.0
        col_mid = (n_col - 1) / 2.0

        # Suppose x = (col - col_mid)*pixel_size, y = (row - row_mid)*pixel_size
        # => M is diag([pixel_size, pixel_size]) but we must be mindful of indexing
        M_2d = torch.eye(2, dtype=torch.float32)
        M_2d[0, 0] = pixel_size  # x scale
        M_2d[1, 1] = pixel_size  # y scale

        b_2d = torch.tensor(
            [-col_mid * pixel_size, -row_mid * pixel_size],
            dtype=torch.float32
        )

        if device is None:
            device = torch.device("cuda" if (backend=="cuda" and torch.cuda.is_available()) else "cpu")

        src_all = src_all.to(device)
        dst_all = dst_all.to(device)
        M_2d = M_2d.to(device)
        b_2d = b_2d.to(device)

        super().__init__(
            n_row=n_row,
            n_col=n_col,
            M=M_2d,
            b=b_2d,
            src=src_all,
            dst=dst_all,
            backend=backend
        )

# -------------------------------------------------------------------
#   Example "uniform" version
# -------------------------------------------------------------------
def build_uniform_static_2d_geometry(
    n_source: int,
    source_radius: float,
    # modules on a circle
    n_module: int,
    module_radius: float,
    det_n_col: int,
    det_spacing: float,
    M_gantry: torch.Tensor,
    b_gantry: torch.Tensor,
    active_sources: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    r"""
    Demo of placing `n_source` sources on a circle and `n_module` modules on another circle,
    each oriented inward. Returns:
      source_positions: [n_source, 2]
      module_centers:   [n_module, 2]
      module_orientations: [n_module, 2,2]
      source_module_mask: [n_source, n_module] (all True for simplicity)
    """
    device = M_gantry.device

    # 1) Source ring
    angles_src = torch.linspace(0, 2*math.pi - 2*math.pi/n_source, n_source).to(device)
    src_list = []
    for theta in angles_src:
        sx = source_radius * math.cos(theta.item())
        sy = source_radius * math.sin(theta.item())
        src_list.append([sx, sy])
    source_positions = torch.tensor(src_list, dtype=torch.float32, device=device)

    # 2) Module ring
    angles_mod = torch.linspace(0, 2*math.pi - 2*math.pi/n_module, n_module).to(device)
    centers_list = []
    orientations_list = []
    for theta in angles_mod:
        cx = module_radius * math.cos(theta.item())
        cy = module_radius * math.sin(theta.item())
        centers_list.append([cx, cy])

        # Inward normal is negative of (cx,cy)
        nx = cx
        ny = cy
        norm_len = math.sqrt(nx*nx + ny*ny)
        if norm_len < 1e-12:
            # fallback
            nx, ny = 1.0, 0.0
            norm_len = 1.0
        nx /= norm_len
        ny /= norm_len

        # We'll define local X axis to run "horizontal" w.r.t. normal, local Y axis = normal
        # For example:
        # normal ~ (nx, ny), define a perpendicular ~ (-ny, nx)
        px = -ny
        py = nx
        # orientation matrix local->gantry => columns = [px, nx; py, ny]
        # If local coords = (u, v), then gantry = Rm*(u, v).
        R = torch.tensor([[px, nx],
                          [py, ny]], dtype=torch.float32, device=device)
        orientations_list.append(R)

    module_centers = torch.tensor(centers_list, dtype=torch.float32, device=device)
    module_orientations = torch.stack(orientations_list, dim=0)  # [n_module, 2,2]

    source_module_mask = torch.ones((n_source, n_module), dtype=torch.bool, device=device)

    return source_positions, module_centers, module_orientations, source_module_mask


class UniformStaticCTProjector2D(StaticCTProjector2D):
    r"""
    A specialized subclass for a simple 2D ring geometry:
      - `n_source` sources on a circle of radius `source_radius`.
      - `n_module` modules on a circle of radius `module_radius`, oriented inward.
      - Each module has `det_n_col` pixels with spacing `det_spacing`.
      - Per‐frame transforms are given by `M_gantry`, `b_gantry`.
      - All sources illuminate all modules (source_module_mask=all True).
    """
    def __init__(
        self,
        n_row: int,
        n_col: int,
        n_source: int,
        source_radius: float,
        n_module: int,
        module_radius: float,
        det_n_col: int,
        det_spacing: float,
        M_gantry: torch.Tensor,
        b_gantry: torch.Tensor,
        active_sources: torch.Tensor,
        pixel_size: float = 1.0,
        backend: str = "cuda"
    ):
        # device = torch.device("cuda" if (backend=="cuda" and torch.cuda.is_available()) else "cpu")
        device = M_gantry.device
        (
            source_positions,
            module_centers,
            module_orientations,
            source_module_mask
        ) = build_uniform_static_2d_geometry(
            n_source=n_source,
            source_radius=source_radius,
            n_module=n_module,
            module_radius=module_radius,
            det_n_col=det_n_col,
            det_spacing=det_spacing,
            M_gantry=M_gantry,
            b_gantry=b_gantry,
            active_sources=active_sources
        )

        super().__init__(
            n_row=n_row,
            n_col=n_col,
            M_gantry=M_gantry.to(device),
            b_gantry=b_gantry.to(device),
            source_positions=source_positions,
            module_centers=module_centers,
            module_orientations=module_orientations,
            det_n_col=det_n_col,
            det_spacing=det_spacing,
            source_module_mask=source_module_mask,
            active_sources=active_sources.to(device),
            pixel_size=pixel_size,
            backend=backend
        )
