import math
import torch
from .ct_projector_2d_module import CTProjector2DModule

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
        backend: str = "cuda"
    ):
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

        # ------------------------------------------------
        # 2) Define an affine transform M, b
        #    We'll interpret (row, col) -> (x, y) as:
        #      x = (col - col_center) * pixel_size
        #      y = (row - row_center) * pixel_size
        #    or some variant. 
        #
        #    Usually, if row is vertical, we'd do 
        #      y = (row_center - row)*pixel_size
        #    but let's keep it simple:
        # ------------------------------------------------
        row_mid = (n_row - 1) / 2.0
        col_mid = (n_col - 1) / 2.0

        # M is effectively the scale from (row, col) to (x, y).
        # But be mindful about row/col <-> x,y ordering.
        # Let's say x = (col)*pixel_size, y = (row)*pixel_size
        # So M = [[0, pixel_size], [pixel_size, 0]] if we want row in y, col in x 
        # but typically we can do a diagonal if row -> y is the same scale, etc.
        # For simplicity, let's do:
        #   x = col * pixel_size
        #   y = row * pixel_size
        #
        # We'll keep an offset so the center is at (0,0).
        # b = [col_mid, row_mid] * pixel_size
        #
        # Actually we only need M in row-col space => (x,y). For orthonormal scaling:
        M = torch.tensor([
            [pixel_size, 0.0],
            [0.0, pixel_size]
        ], dtype=torch.float32)

        b = torch.tensor([-col_mid * pixel_size, -row_mid * pixel_size], dtype=torch.float32)

        # But note we used (col, row) ordering in `b`. 
        # The usage is up to you as long as the intersection code 
        # is consistent. 
        #
        # You can adjust signs as needed if you want the y-axis 
        # to go upward. For example, 
        # b = [col_mid, row_mid] * pixel_size, 
        # and if you want row to go downward, you'd do a negative factor in M[1,1].
        # We'll keep it simple here.

        # Move everything to the same device so we don't get a mismatch
        # in the parent's constructor.
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
