"""Exact chord length of rays through an analytic z-aligned cylinder, optionally clipped to a z slab.

Used as a smooth object basis for calibration fits (e.g. the uniform ACR module: water cylinder of fitted radius, density and
axis position) without a voxel projector.  Differentiable in the cylinder parameters.
"""
import torch


def cylinder_chords(src, dst, cx, cy, radius, z_slabs=((-1e4, 1e4),)):
    """Chord length (mm) of segments src -> dst [..., 3] inside the cylinder (x-cx)^2 + (y-cy)^2 < R^2 for each z slab.

    Returns a list, one tensor [...] per (z_lo, z_hi) in ``z_slabs``.  Rays parallel to z inside the cylinder are treated as
    having a tiny z-slope (no special casing needed for CT geometries).
    """
    d = dst - src
    w = src[..., :2] - torch.stack([cx, cy])
    dp = d[..., :2]
    A = (dp * dp).sum(-1).clamp(min=1e-9)
    B = (w * dp).sum(-1)
    C = (w * w).sum(-1) - radius ** 2
    disc = B * B - A * C
    sq = torch.sqrt(disc.clamp(min=0) + 1e-12)
    t1 = ((-B - sq) / A).clamp(0, 1)
    t2 = ((-B + sq) / A).clamp(0, 1)
    dz = d[..., 2]
    dz = torch.where(dz.abs() < 1e-9, torch.full_like(dz, 1e-9), dz)
    length = d.norm(dim=-1)
    out = []
    for lo, hi in z_slabs:
        ta, tb = (lo - src[..., 2]) / dz, (hi - src[..., 2]) / dz
        seg = (torch.minimum(t2, torch.maximum(ta, tb)) - torch.maximum(t1, torch.minimum(ta, tb))).clamp(min=0)
        out.append(seg * length * (disc > 0))
    return out
