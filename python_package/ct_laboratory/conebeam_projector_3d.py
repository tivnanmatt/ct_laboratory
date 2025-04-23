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
    r"""
    Constructs 3D cone-beam geometry for a standard CT setup.

    - The isocenter is at (cx, cy, cz) = `volume_center_xyz`.
    - For each view θ (in [0..2π)), the source is at radius = `sid` from isocenter
      in the XY plane, with Z kept at `cz`.
    - The detector center is at radius = (sdd - sid) behind the isocenter, also in XY plane.
    - The detector plane is oriented such that:
        - One axis is perpendicular to the ray from source to detector-center (in XY).
        - The other axis is the Z direction.

    Parameters
    ----------
    n_view : int
        Number of projection angles.
    det_nx : int
        Number of detector pixels in the "horizontal" direction (U-axis).
    det_ny : int
        Number of detector pixels in the "vertical" direction (V-axis).
    sid : float
        Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    det_spacing : float
        Physical spacing of detector pixels.
    angle_offset : float
        Offset of the initial rotation angle in radians.
    volume_center_xyz : (float, float, float)
        Center of the volume (the isocenter) in world coordinates.

    Returns
    -------
    src : torch.Tensor, shape = [n_view * det_nx * det_ny, 3]
    dst : torch.Tensor, shape = [n_view * det_nx * det_ny, 3]
    """
    cx, cy, cz = volume_center_xyz
    angles = torch.linspace(0, 2 * math.pi -  2 * math.pi/n_view, n_view) + angle_offset

    all_src = []
    all_dst = []

    for theta in angles:
        # Source position
        sx = cx + sid * math.cos(theta)
        sy = cy + sid * math.sin(theta)
        sz = cz  # Keep the source in plane z=cz

        # Detector center
        dx_c = cx - (sdd - sid) * math.cos(theta)
        dy_c = cy - (sdd - sid) * math.sin(theta)
        dz_c = cz

        # The "horizontal" axis is perpendicular in XY.
        perp_x = -(dy_c - sy)
        perp_y = (dx_c - sx)
        perp_z = 0.0
        norm_len = math.sqrt(perp_x**2 + perp_y**2)
        if norm_len < 1e-12:
            continue
        perp_x /= norm_len
        perp_y /= norm_len

        # The "vertical" axis is along Z
        v_x, v_y, v_z = 0.0, 0.0, 1.0

        mid_u = (det_nx - 1) / 2.0
        mid_v = (det_ny - 1) / 2.0

        for iu in range(det_nx):
            for iv in range(det_ny):
                offset_u = (iu - mid_u) * det_spacing
                offset_v = (iv - mid_v) * det_spacing
                cell_x = dx_c + offset_u * perp_x + offset_v * v_x
                cell_y = dy_c + offset_u * perp_y + offset_v * v_y
                cell_z = dz_c + offset_u * perp_z + offset_v * v_z

                all_src.append([sx, sy, sz])
                all_dst.append([cell_x, cell_y, cell_z])

    src = torch.tensor(all_src, dtype=torch.float32)
    dst = torch.tensor(all_dst, dtype=torch.float32)
    return src, dst


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
        backend: str = "cuda"
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
            backend=backend
        )
