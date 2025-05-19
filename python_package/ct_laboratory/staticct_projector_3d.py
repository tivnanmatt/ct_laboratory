import math
import torch
from typing import Optional
from .ct_projector_3d_module import CTProjector3DModule



def build_circular_sequence(
        n_frame: int,
        n_source: int,
        device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M_list = []
    b_list = []
    active_sources = torch.zeros(n_frame, n_source, dtype=torch.bool, device=device)
    for i in range(n_frame):
        M_ = torch.eye(3, dtype=torch.float32, device=device)   # 3x3 identity (no rotation)
        b_ = torch.zeros(3, dtype=torch.float32, device=device)   # zero offset
        M_list.append(M_)
        b_list.append(b_)
        active_sources[i, i % n_source] = True  # frame i activates source i
    M_gantry = torch.stack(M_list, dim=0)  # shape: [n_frame, 3, 3]
    b_gantry = torch.stack(b_list, dim=0)

    return M_gantry, b_gantry, active_sources


def build_helical_sequence(
        n_frame: int,
        n_source: int,
        pitch: float,
        device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    translation_per_frame = pitch / n_source
    translation_z_offset = n_frame * translation_per_frame / 2

    M_list = []
    b_list = []
    active_sources = torch.zeros(n_frame, n_source, dtype=torch.bool, device=device)
    for i in range(n_frame):
        M_ = torch.eye(3, dtype=torch.float32, device=device)   # 3x3 identity (no rotation)
        b_ = torch.zeros(3, dtype=torch.float32, device=device)   # zero offset
        b_[2] = i * translation_per_frame - translation_z_offset # z-offset
        M_list.append(M_)
        b_list.append(b_)
        active_sources[i, i % n_source] = True  # frame i activates source i
    M_gantry = torch.stack(M_list, dim=0)  # shape: [n_frame, 3, 3]
    b_gantry = torch.stack(b_list, dim=0)

    return M_gantry, b_gantry, active_sources




def build_static_3d_geometry(
    # Basic inputs describing the static gantry geometry
    source_positions: torch.Tensor,  # [n_source, 3] in gantry coords
    module_centers: torch.Tensor,    # [n_module, 3] in gantry coords
    module_orientations: torch.Tensor,  # [n_module, 3, 3], local -> gantry
    det_nx_per_module: int,
    det_ny_per_module: int,
    det_spacing_x: float,
    det_spacing_y: float,
    source_module_mask: torch.Tensor,  # [n_source, n_module] boolean
    # Frame-dependent gantry transforms
    M_gantry: torch.Tensor,  # [n_frame, 3, 3]
    b_gantry: torch.Tensor,  # [n_frame, 3]
    active_sources: torch.Tensor,  # [n_frame, n_source] boolean
    device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Build the full set of (src,dst) rays given:
      - A static configuration of `n_source` sources and `n_module` modules in *gantry coords*.
      - Per-frame transforms M_gantry[i], b_gantry[i] that map gantry coords -> final (x,y,z).
      - A boolean mask active_sources[i,s] for which sources are active on frame i.
      - A boolean source_module_mask[s,m] for which modules are illuminated by a given source.

    Returns
    -------
    src_out : torch.Tensor, shape = [N, 3]
    dst_out : torch.Tensor, shape = [N, 3]
        Where N = total number of active rays.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_frame = M_gantry.shape[0]
    n_source = source_positions.shape[0]
    n_module = module_centers.shape[0]

    # Number of detector pixels per module = det_nx_per_module * det_ny_per_module
    pixel_positions_local = []
    mid_u = (det_nx_per_module - 1) / 2.0
    mid_v = (det_ny_per_module - 1) / 2.0

    for iu in range(det_nx_per_module):
        for iv in range(det_ny_per_module):
            offset_u = (iu - mid_u) * det_spacing_x
            offset_v = (iv - mid_v) * det_spacing_y
            # local (u, v, 0)
            pixel_positions_local.append([offset_u, offset_v, 0.0])
    pixel_positions_local = torch.tensor(pixel_positions_local, dtype=torch.float32, device=device)

    all_src = []
    all_dst = []

    for i_frame in range(n_frame):
        Mf = M_gantry[i_frame]  # [3,3]
        bf = b_gantry[i_frame]  # [3]

        active_src_mask = active_sources[i_frame]  # shape [n_source], bool

        for s_idx in range(n_source):
            if not bool(active_src_mask[s_idx]):
                continue

            src_gantry = source_positions[s_idx]  # [3]
            # source in final coords
            src_image = Mf @ src_gantry + bf

            for m_idx in range(n_module):
                if not bool(source_module_mask[s_idx, m_idx]):
                    continue

                center_gantry = module_centers[m_idx]       # [3]
                Rm = module_orientations[m_idx]             # [3,3]

                # local -> gantry => Rm*(u,v,0) + center
                pixels_gantry = pixel_positions_local @ Rm.transpose(0, 1)
                pixels_gantry += center_gantry

                # gantry -> final coords
                pixels_image = pixels_gantry @ Mf.transpose(0, 1) + bf

                # Expand source
                src_batch = src_image.unsqueeze(0).expand_as(pixels_image)

                all_src.append(src_batch)
                all_dst.append(pixels_image)

    if len(all_src) == 0:
        src_out = torch.empty((0, 3), dtype=torch.float32, device=device)
        dst_out = torch.empty((0, 3), dtype=torch.float32, device=device)
        return src_out, dst_out

    src_out = torch.cat(all_src, dim=0)
    dst_out = torch.cat(all_dst, dim=0)
    return src_out, dst_out


class StaticCTProjector3D(CTProjector3DModule):
    r"""
    A general static gantry 3D projector that extends CTProjector3DModule,
    now using explicit M,b for (i,j,k)->(x,y,z).

    The user provides:
      - Volume size (n_x, n_y, n_z)
      - M_gantry, b_gantry for per-frame transforms (gantry->final coords)
      - source_positions: shape [n_source,3]
      - module_centers, module_orientations
      - det_nx_per_module, det_ny_per_module, spacing
      - source_module_mask, active_sources
      - M, b: transform from (i,j,k)->(x,y,z) for the parent's intersection-based approach
    """
    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_z: int,
        M_gantry: torch.Tensor,       # [n_frame, 3,3]
        b_gantry: torch.Tensor,       # [n_frame, 3]
        source_positions: torch.Tensor,       # [n_source, 3] (gantry coords)
        module_centers: torch.Tensor,         # [n_module, 3]
        module_orientations: torch.Tensor,    # [n_module, 3,3]
        det_nx_per_module: int,
        det_ny_per_module: int,
        det_spacing_x: float,
        det_spacing_y: float,
        source_module_mask: torch.Tensor,     # [n_source, n_module]
        active_sources: torch.Tensor,         # [n_frame, n_source]
        M: torch.Tensor,                      # [3,3], transform from (i,j,k)->(x,y,z)
        b: torch.Tensor,                      # [3]
        backend: str = "cuda",
        device: Optional[torch.device] = None,
        precomputed_intersections: bool = False
    ):
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # 1) Build final (src,dst) for all frames in final coords.
        src_all, dst_all = build_static_3d_geometry(
            source_positions=source_positions,
            module_centers=module_centers,
            module_orientations=module_orientations,
            det_nx_per_module=det_nx_per_module,
            det_ny_per_module=det_ny_per_module,
            det_spacing_x=det_spacing_x,
            det_spacing_y=det_spacing_y,
            source_module_mask=source_module_mask,
            M_gantry=M_gantry,
            b_gantry=b_gantry,
            active_sources=active_sources,
            device=device
        )

        src_all = src_all.to(device)
        dst_all = dst_all.to(device)
        M = M.to(device)
        b = b.to(device)

        super().__init__(
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            M=M,
            b=b,
            src=src_all,
            dst=dst_all,
            backend=backend,
            device=device,
            precomputed_intersections=precomputed_intersections
        )




# def build_MTEC_geometry(
#     n_source: int,
#     source_radius: float,
#     source_z_offset: float,
#     n_module: int,
#     module_radius: float,
#     module_z_offset: float,
#     modules_per_source: int = 16,
#     device: Optional[torch.device] = None
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     r"""
#     Example helper that arranges:
#       - `n_source` sources on a circle (radius=source_radius) in XY plane, with Z=source_z_offset.
#       - `n_module` modules on a ring of radius=module_radius in XY plane at z=module_z_offset,
#         oriented inward, each with local pixel (u,v).
#       - All sources illuminate all modules (source_module_mask=all True).
#     """
    
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     angles_source = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_source, n_source, device=device)

#     # 1) Source positions
#     src_list = []
#     for theta in angles_source:
#         x = source_radius * math.cos(theta.item())
#         y = source_radius * math.sin(theta.item())
#         z = source_z_offset
#         src_list.append([x, y, z])
#     source_positions = torch.tensor(src_list, dtype=torch.float32, device=device)

#     # 2) Module centers
#     angles_module = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_module, n_module, device=device)
#     centers_list = []
#     orientations_list = []
#     for theta in angles_module:
#         cx = module_radius * math.cos(theta.item())
#         cy = module_radius * math.sin(theta.item())
#         cz = module_z_offset
#         centers_list.append([cx, cy, cz])

#         # inward normal = (cx, cy, 0)
#         nx = cx
#         ny = cy
#         nz = 0.0
#         nn = math.sqrt(nx*nx + ny*ny + nz*nz)
#         if nn < 1e-12:
#             nx, ny, nz = 1.0, 0.0, 0.0
#             nn = 1.0
#         nx /= nn
#         ny /= nn
#         nz /= nn

#         # 'up' direction = z-axis => (0,0,1)
#         up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
#         normal = torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
#         side = torch.cross(up, normal)
#         side_len = side.norm()
#         if side_len < 1e-12:
#             side = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
#             side_len = 1.0
#         side = side / side_len
#         up_ortho = torch.cross(normal, side)
#         # orientation matrix local->gantry => columns = [side, up_ortho, normal]
#         R = torch.stack([side, up_ortho, normal], dim=1)
#         orientations_list.append(R)

#     module_centers = torch.tensor(centers_list, dtype=torch.float32, device=device)
#     module_orientations = torch.stack(orientations_list, dim=0)


#     source_module_mask = torch.zeros((n_source, n_module), dtype=torch.bool, device=device)
#     for i in range(n_source):
#         module_distances = torch.norm(module_centers - source_positions[i], dim=1)
#         # find the indices of the farthest modules
#         _, farthest_indices = torch.topk(module_distances, modules_per_source, largest=True)
#         # set the rest to False
#         source_module_mask[i, :] = False
#         source_module_mask[i, farthest_indices] = True



#     return source_positions, module_centers, module_orientations, source_module_mask





# def build_MTEC_geometry(
#     n_source: int,
#     source_radius: float,
#     source_z_offset: float,
#     n_module: int,
#     module_radius: float,
#     module_z_offset: float,
#     modules_per_source: int = 16,
#     device: Optional[torch.device] = None
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

def build_uniform_static_3d_geometry(
    n_source: int,
    source_radius: float,
    source_z_offset: float,
    n_module: int,
    module_radius: float,
    module_z_offset: float,
    modules_per_source: int,
    device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Example helper that arranges:
      - `n_source` sources on a circle (radius=source_radius) in XY plane, with Z=source_z_offset.
      - `n_module` modules on a ring of radius=module_radius in XY plane at z=module_z_offset,
        oriented inward, each with local pixel (u,v).
      - All sources illuminate all modules (source_module_mask=all True).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    angles_source = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_source, n_source, device=device)

    # 1) Source positions
    src_list = []
    for theta in angles_source:
        x = source_radius * math.cos(theta.item())
        y = source_radius * math.sin(theta.item())
        z = source_z_offset
        src_list.append([x, y, z])
    source_positions = torch.tensor(src_list, dtype=torch.float32, device=device)

    # 2) Module centers
    angles_module = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_module, n_module, device=device)
    centers_list = []
    orientations_list = []
    for theta in angles_module:
        cx = module_radius * math.cos(theta.item())
        cy = module_radius * math.sin(theta.item())
        cz = module_z_offset
        centers_list.append([cx, cy, cz])

        # inward normal = (cx, cy, 0)
        nx = cx
        ny = cy
        nz = 0.0
        nn = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nn < 1e-12:
            nx, ny, nz = 1.0, 0.0, 0.0
            nn = 1.0
        nx /= nn
        ny /= nn
        nz /= nn

        # 'up' direction = z-axis => (0,0,1)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        normal = torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
        side = torch.cross(up, normal)
        side_len = side.norm()
        if side_len < 1e-12:
            side = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
            side_len = 1.0
        side = side / side_len
        up_ortho = torch.cross(normal, side)
        # orientation matrix local->gantry => columns = [side, up_ortho, normal]
        R = torch.stack([side, up_ortho, normal], dim=1)
        orientations_list.append(R)

    module_centers = torch.tensor(centers_list, dtype=torch.float32, device=device)
    module_orientations = torch.stack(orientations_list, dim=0)


    # source_module_mask = torch.ones((n_source, n_module), dtype=torch.bool, device=device)

    source_module_mask = torch.zeros((n_source, n_module), dtype=torch.bool, device=device)
    for i in range(n_source):
        module_distances = torch.norm(module_centers - source_positions[i], dim=1)
        # find the indices of the farthest modules
        _, farthest_indices = torch.topk(module_distances, modules_per_source, largest=True)
        # set the rest to False
        source_module_mask[i, :] = False
        source_module_mask[i, farthest_indices] = True

    return source_positions, module_centers, module_orientations, source_module_mask


class UniformStaticCTProjector3D(StaticCTProjector3D):
    r"""
    A convenience subclass that builds a uniform ring geometry in 3D:
      - `n_source` sources uniformly around a ring at radius=`source_radius`, z=`source_z_offset`
      - `n_module` modules on a ring radius=`module_radius`, z=`module_z_offset`, oriented inward
      - Each module has (det_nx_per_module x det_ny_per_module) pixels
      - M_gantry, b_gantry map gantry coords -> final coords
      - `M, b` is the transform from (i,j,k)->(x,y,z) for the parent's intersection logic
    """
    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_z: int,
        n_source: int,
        source_radius: float,
        source_z_offset: float,
        n_module: int,
        module_radius: float,
        det_nx_per_module: int,
        det_ny_per_module: int,
        det_spacing_x: float,
        det_spacing_y: float,
        module_z_offset: float,
        M_gantry: torch.Tensor,
        b_gantry: torch.Tensor,
        active_sources: torch.Tensor,
        modules_per_source: int,
        M: torch.Tensor,  # [3,3], (i,j,k)->(x,y,z)
        b: torch.Tensor,  # [3]
        backend: str = "cuda",
        precomputed_intersections=False,
        device: Optional[torch.device] = None
    ):
        
        if device is None:
            device = torch.device("cuda" if backend == "cuda" and torch.cuda.is_available() else "cpu")

        (
            source_positions,
            module_centers,
            module_orientations,
            source_module_mask
        ) = build_uniform_static_3d_geometry(
            n_source=n_source,
            source_radius=source_radius,
            source_z_offset=source_z_offset,
            n_module=n_module,
            module_radius=module_radius,
            module_z_offset=module_z_offset,
            modules_per_source=modules_per_source,
            device=device
        )

        super().__init__(
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            M_gantry=M_gantry.to(device),
            b_gantry=b_gantry.to(device),
            source_positions=source_positions,
            module_centers=module_centers,
            module_orientations=module_orientations,
            det_nx_per_module=det_nx_per_module,
            det_ny_per_module=det_ny_per_module,
            det_spacing_x=det_spacing_x,
            det_spacing_y=det_spacing_y,
            source_module_mask=source_module_mask,
            active_sources=active_sources.to(device),
            M=M.to(device),
            b=b.to(device),
            backend=backend,
            precomputed_intersections=precomputed_intersections,
            device=device
        )
