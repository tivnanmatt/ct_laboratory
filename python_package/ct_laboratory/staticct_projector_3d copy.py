import math
import torch
from typing import Optional
from .ct_projector_3d_module import CTProjector3DModule

def build_static_3d_geometry(
    # Basic inputs describing the static gantry geometry
    source_positions: torch.Tensor,  # [n_source, 3] in gantry coords
    module_centers: torch.Tensor,    # [n_module, 3] in gantry coords
    module_orientations: torch.Tensor,  # [n_module, 3, 3], each is rotation from local pixel coords to gantry coords
    det_nx_per_module: int,
    det_ny_per_module: int,
    det_spacing_x: float,
    det_spacing_y: float,
    source_module_mask: torch.Tensor,  # [n_source, n_module] boolean
    # Frame-dependent gantry transformations
    M_gantry: torch.Tensor,  # [n_frame, 3, 3]
    b_gantry: torch.Tensor,  # [n_frame, 3]
    active_sources: torch.Tensor,  # [n_frame, n_source] boolean
) -> (torch.Tensor, torch.Tensor):
    r"""
    Build the full set of (src,dst) rays given:
      - A static configuration of `n_source` sources and `n_module` detector modules in *gantry coordinates*.
      - For each module, the local detector grid size and pixel spacing, as well as a 3×3 orientation matrix that maps local pixel (u,v) axes into gantry coords.
      - A per-frame affine transform M_gantry[i], b_gantry[i] that maps gantry coords -> image coords.
      - A boolean mask active_sources[i,s] indicating which sources are active on frame i.
      - A boolean source_module_mask[s,m] indicating whether source s illuminates module m.

    Returns
    -------
    src : torch.Tensor
        Shape: [N, 3], where N = total number of rays across all frames, sources, modules, and pixels.
    dst : torch.Tensor
        Same shape, listing the corresponding detector pixel coordinates in the image domain.
    """

    device = source_positions.device
    n_frame = M_gantry.shape[0]
    n_source = source_positions.shape[0]
    n_module = module_centers.shape[0]

    # Number of detector pixels per module = det_nx_per_module * det_ny_per_module
    # We'll generate all pixel offsets in local module coords, then transform to gantry coords.

    # Precompute local pixel coordinate offsets for each module's grid
    # shape => [det_nx_per_module * det_ny_per_module, 3] (the third coordinate can be 0 if purely 2D in-plane)
    pixel_positions_local = []
    mid_u = (det_nx_per_module - 1) / 2.0
    mid_v = (det_ny_per_module - 1) / 2.0

    for iu in range(det_nx_per_module):
        for iv in range(det_ny_per_module):
            offset_u = (iu - mid_u) * det_spacing_x
            offset_v = (iv - mid_v) * det_spacing_y
            # local detector plane => (u, v, 0)
            pixel_positions_local.append([offset_u, offset_v, 0.0])

    pixel_positions_local = torch.tensor(pixel_positions_local, dtype=torch.float32, device=device)  # [P,3]
    n_pixel = pixel_positions_local.shape[0]

    # We'll accumulate all final (src, dst) in lists, then cat at the end
    all_src = []
    all_dst = []

    for i_frame in range(n_frame):
        # 1) Extract this frame's M,b
        Mf = M_gantry[i_frame]       # [3,3]
        bf = b_gantry[i_frame]       # [3]
        # 2) Identify which sources are active in this frame
        active_src_mask = active_sources[i_frame]  # shape [n_source], boolean

        # For each source
        for s_idx in range(n_source):
            if not bool(active_src_mask[s_idx]):
                continue  # skip inactive sources

            # 3) The source position in gantry coords => transform to image coords
            src_gantry = source_positions[s_idx]  # [3]
            src_image = Mf @ src_gantry + bf      # shape [3]

            # For each module that is illuminated by this source
            for m_idx in range(n_module):
                if not bool(source_module_mask[s_idx, m_idx]):
                    continue

                # module center + orientation in gantry coords
                center_gantry = module_centers[m_idx]                 # [3]
                Rm = module_orientations[m_idx]                       # [3,3], local->gantry
                # Transform local pixel coords => gantry
                # pixel_gantry = Rm @ (u,v,0) + center_gantry
                # We'll do it for each pixel below

                # Then apply M_gantry to get image coords
                # pixel_image = Mf @ pixel_gantry + bf

                # For performance, we can do it in a batched manner
                # local_pixels shape => [P,3]
                pixels_gantry = pixel_positions_local @ Rm.transpose(0, 1)  # [P,3] local->gantry
                pixels_gantry += center_gantry  # broadcast add => [P,3]
                # Now map gantry->image
                pixels_image = pixels_gantry @ Mf.transpose(0, 1) + bf  # [P,3]

                # Create repeated src array => [P,3]
                src_batch = src_image.unsqueeze(0).expand_as(pixels_image)

                all_src.append(src_batch)
                all_dst.append(pixels_image)

    # Concatenate everything
    if len(all_src) == 0:
        # if no rays are active, return empty
        src_out = torch.empty((0,3), dtype=torch.float32, device=device)
        dst_out = torch.empty((0,3), dtype=torch.float32, device=device)
        return src_out, dst_out

    src_out = torch.cat(all_src, dim=0)  # shape [N,3]
    dst_out = torch.cat(all_dst, dim=0)  # shape [N,3]
    return src_out, dst_out


class StaticCTProjector3D(CTProjector3DModule):
    r"""
    A general static gantry 3D projector that extends CTProjector3DModule.

    The user provides:
      - Volume size (n_x, n_y, n_z)
      - M_gantry and b_gantry: per-frame affine transforms from gantry coords -> image coords
      - source_positions: shape [n_source,3] in gantry coords
      - module_centers, module_orientations, pixel spacing, etc.
      - source_module_mask: shape [n_source, n_module]
      - active_sources: shape [n_frame, n_source]
      - Additional geometry details for the modules.

    This class will build one big set of (src,dst) rays for all frames,
    then pass them to the parent CTProjector3DModule for the intersection‐based
    forward/back projection.
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
        module_orientations: torch.Tensor,    # [n_module, 3, 3]
        det_nx_per_module: int,
        det_ny_per_module: int,
        det_spacing_x: float,
        det_spacing_y: float,
        source_module_mask: torch.Tensor,     # [n_source, n_module] boolean
        active_sources: torch.Tensor,         # [n_frame, n_source] boolean
        voxel_size: float = 1.0,
        backend: str = "cuda"
    ):
        # 1) Build final (src,dst) for all frames
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
        )

        # 2) Construct affine transform for (i,j,k)->(x,y,z)
        i_mid = (n_x - 1) / 2.0
        j_mid = (n_y - 1) / 2.0
        k_mid = (n_z - 1) / 2.0

        M_world = torch.eye(3, dtype=torch.float32)
        M_world[0, 0] = voxel_size
        M_world[1, 1] = voxel_size
        M_world[2, 2] = voxel_size

        b_world = torch.tensor(
            [-i_mid * voxel_size, -j_mid * voxel_size, -k_mid * voxel_size],
            dtype=torch.float32
        )

        device = torch.device("cuda" if backend == "cuda" and torch.cuda.is_available() else "cpu")
        src_all = src_all.to(device)
        dst_all = dst_all.to(device)
        M_world = M_world.to(device)
        b_world = b_world.to(device)

        super().__init__(
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            M=M_world,
            b=b_world,
            src=src_all,
            dst=dst_all,
            backend=backend
        )

def build_uniform_static_3d_geometry(
    # Uniform source ring
    n_source: int,
    source_radius: float,
    source_z_offset: float,
    # Uniform modules on a polygon
    n_module: int,
    module_radius: float,   # <-- added parameter!
    det_nx_per_module: int,
    det_ny_per_module: int,
    det_spacing_x: float,
    det_spacing_y: float,
    module_z_offset: float,
    # Frame transforms
    M_gantry: torch.Tensor,
    b_gantry: torch.Tensor,
    active_sources: torch.Tensor,
) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    r"""
    Example helper that arranges:
      - n_source sources on a circle (radius=source_radius) in XY plane, with Z=source_z_offset.
      - n_module modules arranged on a regular polygon in XY plane at a distance module_radius,
        oriented to face inward.
    
    Returns:
      source_positions: [n_source,3] in gantry coords
      module_centers: [n_module,3] in gantry coords
      module_orientations: [n_module,3,3]
      source_module_mask: [n_source, n_module] (all True for simplicity)
    """
    device = M_gantry.device
    angles_source = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_source, n_source).to(device)

    # 1) Compute source positions on a ring
    src_list = []
    for theta in angles_source:
        x = source_radius * math.cos(theta.item())
        y = source_radius * math.sin(theta.item())
        z = source_z_offset
        src_list.append([x, y, z])
    source_positions = torch.tensor(src_list, dtype=torch.float32, device=device)

    # 2) Compute module centers on a polygon using the provided module_radius.
    angles_module = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_module, n_module).to(device)
    centers_list = []
    orientations_list = []
    for theta in angles_module:
        cx = module_radius * math.cos(theta.item())
        cy = module_radius * math.sin(theta.item())
        cz = module_z_offset
        centers_list.append([cx, cy, cz])

        # Orientation: we want the module normal to point inward.
        # Use the XY components to determine the inward direction.
        nx = cx
        ny = cy
        nz = 0.0
        nn = math.sqrt(nx * nx + ny * ny + nz * nz)
        if nn < 1e-12:
            nx, ny, nz = 1.0, 0.0, 0.0
            nn = 1.0
        nx /= nn; ny /= nn; nz /= nn

        # Define the 'up' direction as the z-axis.
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        normal = torch.tensor([nx, ny, nz], dtype=torch.float32, device=device)
        side = torch.cross(up, normal)
        side_norm = side.norm()
        if side_norm < 1e-12:
            side = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
            side_norm = 1.0
        side /= side_norm
        # Recalculate the orthogonal up vector
        up_ortho = torch.cross(normal, side)
        # Orientation matrix: local X = side, local Y = up_ortho, local Z = normal
        R = torch.stack([side, up_ortho, normal], dim=1)
        orientations_list.append(R)

    module_centers = torch.tensor(centers_list, dtype=torch.float32, device=device)
    module_orientations = torch.stack(orientations_list, dim=0)
    source_module_mask = torch.ones((n_source, n_module), dtype=torch.bool, device=device)

    return source_positions, module_centers, module_orientations, source_module_mask


class UniformStaticCTProjector3D(StaticCTProjector3D):
    r"""
    A convenience subclass of StaticCTProjector3D that builds a uniform geometry:
      - n_source sources uniformly distributed on a ring in the XY plane
        (radius=source_radius, z offset = source_z_offset).
      - n_module detector modules arranged on a regular polygon in the XY plane,
        with centers on a ring of radius module_radius (z offset = module_z_offset)
        and oriented inward.
      - Each module has a grid of shape (det_nx_per_module, det_ny_per_module) with
        spacings (det_spacing_x, det_spacing_y).
      - Per‐frame transforms M_gantry, b_gantry map gantry coordinates to image coordinates.
      - active_sources indicates which sources are active in each frame.
    """
    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_z: int,
        # Sources
        n_source: int,
        source_radius: float,
        source_z_offset: float,
        # Modules
        n_module: int,
        module_radius: float,  # <-- added parameter!
        det_nx_per_module: int,
        det_ny_per_module: int,
        det_spacing_x: float,
        det_spacing_y: float,
        module_z_offset: float,
        # Gantry transforms + active
        M_gantry: torch.Tensor,
        b_gantry: torch.Tensor,
        active_sources: torch.Tensor,
        voxel_size: float = 1.0,
        backend: str = "cuda"
    ):
        device = torch.device("cuda" if backend == "cuda" and torch.cuda.is_available() else "cpu")

        # Build uniform geometry in gantry coordinates using the given module_radius.
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
            module_radius=module_radius,  # use the passed module_radius
            det_nx_per_module=det_nx_per_module,
            det_ny_per_module=det_ny_per_module,
            det_spacing_x=det_spacing_x,
            det_spacing_y=det_spacing_y,
            module_z_offset=module_z_offset,
            M_gantry=M_gantry,
            b_gantry=b_gantry,
            active_sources=active_sources
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
            voxel_size=voxel_size,
            backend=backend
        )