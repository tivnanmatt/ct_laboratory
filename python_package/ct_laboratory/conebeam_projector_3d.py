import math
import torch
from .ct_projector_3d_module import CTProjector3DModule










def build_conebeam_3d_geometry(
    n_view: int,
    det_nx: int,
    det_ny: int,
    sid: float,          # Source-to-Isocenter Distance
    sdd: float,          # Source-to-Detector Distance
    det_spacing: float = 1.0,
    angle_offset: float = 0.0,
    volume_center_xyz=(0.0, 0.0, 0.0),
):
    """
    Vectorized 3D cone-beam geometry:
    - src: [n_view * det_ny * det_nx, 3]
    - dst: [n_view * det_ny * det_nx, 3]
    """
    cx, cy, cz = volume_center_xyz

    # 1) angles & source / detector-center positions
    angles = torch.linspace(0,
                            2*math.pi - 2*math.pi/n_view,
                            n_view,
                            dtype=torch.float32)
    angles = angles + angle_offset

    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)

    # source pos: [n_view]
    sx = cx + sid * cos_t
    sy = cy + sid * sin_t
    sz = torch.full_like(sx, cz)

    # detector center pos: [n_view]
    dx_c = cx - (sdd - sid) * cos_t
    dy_c = cy - (sdd - sid) * sin_t
    dz_c = torch.full_like(dx_c, cz)

    # 2) build perp-axis in XY (⊥ ray direction)
    #    (dy_c - sy) = -sdd*sin,  (dx_c - sx) = -sdd*cos
    perp_x = -(dy_c - sy)
    perp_y =  (dx_c - sx)
    # normalize
    norm = torch.sqrt(perp_x*perp_x + perp_y*perp_y)
    perp_x = perp_x / norm
    perp_y = perp_y / norm
    perp_z = torch.zeros_like(perp_x)

    # 3) detector pixel offsets
    mid_u = (det_nx - 1) / 2.0
    mid_v = (det_ny - 1) / 2.0

    u = (torch.arange(det_nx, dtype=torch.float32) - mid_u) * det_spacing  # [det_nx]
    v = (torch.arange(det_ny, dtype=torch.float32) - mid_v) * det_spacing  # [det_ny]

    # make meshgrid of offsets: U, V are [det_ny, det_nx]
    U = u.unsqueeze(0).expand(det_ny, det_nx)
    V = v.unsqueeze(1).expand(det_ny, det_nx)

    # 4) reshape all view-dependent quantities to [n_view,1,1]
    shape = (n_view, 1, 1)
    sx = sx.view(shape)
    sy = sy.view(shape)
    sz = sz.view(shape)
    dx_c = dx_c.view(shape)
    dy_c = dy_c.view(shape)
    dz_c = dz_c.view(shape)
    perp_x = perp_x.view(shape)
    perp_y = perp_y.view(shape)
    perp_z = perp_z.view(shape)

    # 5) compute the detector cell coordinates for every view & pixel
    #    each is [n_view, det_ny, det_nx]
    cell_x = dx_c + perp_x * U +      0.0 * V  # v_x=0
    cell_y = dy_c + perp_y * U +      0.0 * V  # v_y=0
    cell_z = dz_c + perp_z * U + 1.0 * V       # v_z=1

    # 6) pack source and detector arrays and flatten
    #    src_grid: [n_view,det_ny,det_nx,3]
    src_grid = torch.stack([
        sx.expand(-1, det_ny, det_nx),
        sy.expand(-1, det_ny, det_nx),
        sz.expand(-1, det_ny, det_nx),
    ], dim=-1)

    dst_grid = torch.stack([cell_x, cell_y, cell_z], dim=-1)

    # flatten to [n_view * det_ny * det_nx, 3]
    src = src_grid.reshape(-1, 3)
    dst = dst_grid.reshape(-1, 3)

    return src, dst






# def build_conebeam_3d_geometry(
#     n_view: int,
#     det_nx: int,
#     det_ny: int,
#     sid: float,          # Source-to-Isocenter Distance
#     sdd: float,          # Source-to-Detector Distance
#     det_spacing: float = 1.0,
#     angle_offset: float = 0.0,
#     volume_center_xyz=(0.0, 0.0, 0.0),
# ):
#     r"""
#     Constructs 3D cone-beam geometry for a standard CT setup.

#     - The isocenter is at (cx, cy, cz) = `volume_center_xyz`.
#     - For each view θ (in [0..2π)), the source is at radius = `sid` from isocenter
#       in the XY plane, with Z kept at `cz`.
#     - The detector center is at radius = (sdd - sid) behind the isocenter, also in XY plane.
#     - The detector plane is oriented such that:
#         - One axis is perpendicular to the ray from source to detector-center (in XY).
#         - The other axis is the Z direction.

#     Parameters
#     ----------
#     n_view : int
#         Number of projection angles.
#     det_nx : int
#         Number of detector pixels in the "horizontal" direction (U-axis).
#     det_ny : int
#         Number of detector pixels in the "vertical" direction (V-axis).
#     sid : float
#         Source-to-Isocenter Distance.
#     sdd : float
#         Source-to-Detector Distance.
#     det_spacing : float
#         Physical spacing of detector pixels.
#     angle_offset : float
#         Offset of the initial rotation angle in radians.
#     volume_center_xyz : (float, float, float)
#         Center of the volume (the isocenter) in world coordinates.

#     Returns
#     -------
#     src : torch.Tensor, shape = [n_view * det_nx * det_ny, 3]
#     dst : torch.Tensor, shape = [n_view * det_nx * det_ny, 3]
#     """
#     cx, cy, cz = volume_center_xyz
#     angles = torch.linspace(0, 2 * math.pi -  2 * math.pi/n_view, n_view) + angle_offset

#     all_src = []
#     all_dst = []

#     for theta in angles:

#         print(f"DEBUG: Angle: {theta:.2f} rad")

#         # Source position
#         sx = cx + sid * math.cos(theta)
#         sy = cy + sid * math.sin(theta)
#         sz = cz  # Keep the source in plane z=cz

#         # Detector center
#         dx_c = cx - (sdd - sid) * math.cos(theta)
#         dy_c = cy - (sdd - sid) * math.sin(theta)
#         dz_c = cz

#         # The "horizontal" axis is perpendicular in XY.
#         perp_x = -(dy_c - sy)
#         perp_y = (dx_c - sx)
#         perp_z = 0.0
#         norm_len = math.sqrt(perp_x**2 + perp_y**2)
#         if norm_len < 1e-12:
#             continue
#         perp_x /= norm_len
#         perp_y /= norm_len

#         # The "vertical" axis is along Z
#         v_x, v_y, v_z = 0.0, 0.0, 1.0

#         mid_u = (det_nx - 1) / 2.0
#         mid_v = (det_ny - 1) / 2.0

#         for iu in range(det_nx):
#             for iv in range(det_ny):
#                 offset_u = (iu - mid_u) * det_spacing
#                 offset_v = (iv - mid_v) * det_spacing
#                 cell_x = dx_c + offset_u * perp_x + offset_v * v_x
#                 cell_y = dy_c + offset_u * perp_y + offset_v * v_y
#                 cell_z = dz_c + offset_u * perp_z + offset_v * v_z

#                 all_src.append([sx, sy, sz])
#                 all_dst.append([cell_x, cell_y, cell_z])

#     src = torch.tensor(all_src, dtype=torch.float32)
#     dst = torch.tensor(all_dst, dtype=torch.float32)
#     return src, dst


class ConeBeam3DProjector(CTProjector3DModule):
    r"""
    A convenience subclass of CTProjector3DModule for standard 3D cone-beam geometry.

    Instead of manually passing `src` and `dst`, the user provides:
      - volume size (n_x, n_y, n_z)
      - number of views (n_view)
      - detector grid size (det_nx, det_ny)
      - SID (source-to-isocenter distance)
      - SDD (source-to-detector distance)
      - voxel_size (to define the transform from (i,j,k) -> (x,y,z))
      - etc.
    """
    def __init__(
        self,
        n_x: int,
        n_y: int,
        n_z: int,
        n_view: int,
        det_nx: int,
        det_ny: int,
        sid: float,
        sdd: float,
        det_spacing: float = 1.0,
        voxel_size: float = 1.0,
        angle_offset: float = 0.0,
        backend: str = "cuda",
        precomputed_intersections: bool = False,
    ):
        # 1) Compute src/dst
        src, dst = build_conebeam_3d_geometry(
            n_view=n_view,
            det_nx=det_nx,
            det_ny=det_ny,
            sid=sid,
            sdd=sdd,
            det_spacing=det_spacing,
            angle_offset=angle_offset,
            volume_center_xyz=(0.0, 0.0, 0.0),
        )

        # 2) Construct an affine transform M, b for (i,j,k)->(x,y,z).
        #    Let's set the center of the volume at i_mid, j_mid, k_mid, 
        #    and voxel size in x,y,z directions = voxel_size.
        i_mid = (n_x - 1) / 2.0
        j_mid = (n_y - 1) / 2.0
        k_mid = (n_z - 1) / 2.0

        # If we assume x = i * voxel_size, y = j * voxel_size, z = k * voxel_size,
        # all orthonormal, then M is diag(voxel_size, voxel_size, voxel_size).
        M = torch.eye(3, dtype=torch.float32)
        M[0, 0] = voxel_size
        M[1, 1] = voxel_size
        M[2, 2] = voxel_size

        # We want the center (i_mid, j_mid, k_mid) to map to (0,0,0).
        # So we set b accordingly:
        b = torch.tensor([-i_mid * voxel_size, -j_mid * voxel_size, -k_mid * voxel_size],
                         dtype=torch.float32)

        device = torch.device("cuda" if backend == "cuda" and torch.cuda.is_available() else "cpu")
        src = src.to(device)
        dst = dst.to(device)
        M = M.to(device)
        b = b.to(device)

        # 3) Call parent constructor
        super().__init__(
            n_x=n_x,
            n_y=n_y,
            n_z=n_z,
            M=M,
            b=b,
            src=src,
            dst=dst,
            backend=backend,
            precomputed_intersections=precomputed_intersections
        )
