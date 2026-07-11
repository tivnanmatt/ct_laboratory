import math
import torch
from .ct_projector_2d_module import CTProjector2DModule
from .standard_image_transform import standard_image_transform_2d

def build_fanbeam_2d_geometry(
    n_view: int,
    n_det: int,
    sid: float,         # Source-to-Isocenter Distance
    sdd: float,         # Source-to-Detector Distance
    det_spacing: float = 1.0,
    angle_offset: float = 0.0,
    image_center_xy=(0.0, 0.0),
):
    r"""
    Constructs 2D fan-beam geometry for a standard CT setup.

    - We place the isocenter at (cx, cy) = `image_center_xy`.
    - For each view `\theta`, the source is at radius = `sid` from isocenter.
    - The detector center is at radius = `sdd - sid` behind the isocenter,
      on the opposite side from the source.

    Parameters
    ----------
    n_view : int
        Number of projection angles.
    n_det : int
        Number of detector elements (1D).
    sid : float
        Source-to-Isocenter Distance.
    sdd : float
        Source-to-Detector Distance.
    det_spacing : float
        Physical spacing between detector elements.
    angle_offset : float
        Starting angle offset in radians (e.g., 0.0 if you want to start at 0).
    image_center_xy : (float, float)
        (cx, cy) location of isocenter in the global XY coordinate system.

    Returns
    -------
    src : torch.Tensor, shape = [n_view * n_det, 2]
    dst : torch.Tensor, shape = [n_view * n_det, 2]
    """
    cx, cy = image_center_xy
    # For each angle
    angles = torch.linspace(0, 2 * math.pi - 2*math.pi/n_view, n_view) + angle_offset

    all_src = []
    all_dst = []
    for theta in angles:
        # Source
        sx = cx + sid * math.cos(theta)
        sy = cy + sid * math.sin(theta)

        # Detector center
        # Distance from isocenter to the detector center is (sdd - sid)
        dx_c = cx - (sdd - sid) * math.cos(theta)
        dy_c = cy - (sdd - sid) * math.sin(theta)

        # A unit vector perpendicular to the ray from source to detector-center
        perp_x = -(dy_c - sy)
        perp_y = (dx_c - sx)
        norm_len = math.sqrt(perp_x**2 + perp_y**2)
        if norm_len < 1e-12:
            # Degenerate geometry (shouldn't happen normally)
            continue
        perp_x /= norm_len
        perp_y /= norm_len

        # Distribute detector pixels along the perpendicular
        mid = (n_det - 1) / 2.0
        for i_det in range(n_det):
            offset = (i_det - mid) * det_spacing
            cell_x = dx_c + offset * perp_x
            cell_y = dy_c + offset * perp_y
            all_src.append([sx, sy])
            all_dst.append([cell_x, cell_y])

    src = torch.tensor(all_src, dtype=torch.float32)
    dst = torch.tensor(all_dst, dtype=torch.float32)
    return src, dst


class FanBeam2DProjector(CTProjector2DModule):
    r"""
    A convenience subclass of CTProjector2DModule for standard 2D fan-beam geometry.
    
    Instead of manually passing `src` and `dst`, the user provides:
      - image size (n_row, n_col)
      - number of views (n_view)
      - number of detector elements (n_det)
      - SID (source-to-isocenter distance)
      - SDD (source-to-detector distance)
      - pixel_size (used for row/col -> x,y if desired)
      - etc.

    This class computes the geometry internally (src, dst) 
    and passes them up to the parent constructor.
    """
    def __init__(
        self,
        n_row: int,
        n_col: int,
        n_view: int,
        n_det: int,
        sid: float,
        sdd: float,
        det_spacing: float = 1.0,
        pixel_size: float = 1.0,
        angle_offset: float = 0.0,
        backend: str = "cuda",
        device = None
    ):
        
        self.n_row = n_row
        self.n_col = n_col
        self.n_view = n_view
        self.n_det = n_det
        self.sid = sid
        self.sdd = sdd
        self.det_spacing = det_spacing
        self.pixel_size = pixel_size
        self.angle_offset = angle_offset
        self.backend = backend
        self.device = device
        


        # ------------------------------------------------
        # 1) Compute src/dst using a helper function
        # ------------------------------------------------
        src, dst = build_fanbeam_2d_geometry(
            n_view=n_view,
            n_det=n_det,
            sid=sid,
            sdd=sdd,
            det_spacing=det_spacing,
            angle_offset=angle_offset,
            image_center_xy=(0.0, 0.0),
        )

        M, b = standard_image_transform_2d(
            n_x=n_row,
            n_y=n_col,
            s_x=pixel_size,
            s_y=pixel_size
        )

        # b = b*0
        # Move everything to the same device so we don't get a mismatch
        # in the parent's constructor.

        if device is None:
            device = torch.device("cuda" if backend == "cuda" and torch.cuda.is_available() else "cpu")
        
        src = src.to(device)
        dst = dst.to(device)
        M = M.to(device)
        b = b.to(device)

        # ------------------------------------------------
        # 3) Call the parent constructor
        # ------------------------------------------------
        super().__init__(
            n_row=n_row,
            n_col=n_col,
            M=M,
            b=b,
            src=src,
            dst=dst,
            backend=backend
        )
