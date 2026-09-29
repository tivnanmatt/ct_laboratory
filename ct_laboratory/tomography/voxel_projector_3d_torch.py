"""Pure-PyTorch backend for the voxel-driven separable-footprint projector.

Same footprint model, weights and sinogram ordering as ``sf_projector_3d.cu``
(SF-TR: trapezoid across columns x rectangle across rows x chord amplitude),
written with vectorised tensor ops so it runs on ANY device (CPU or GPU) and
serves as an independent reference for the CUDA kernels.  It mirrors the
``backend='torch'`` option of the ray-driven projector: slower than the CUDA
kernels but device-agnostic and easy to read.

Voxels are processed in chunks; views are looped over one at a time.  All the
footprint arithmetic is done in float64 and results are returned in float32.
"""
from __future__ import annotations

import math

import torch

from .voxel_projector_3d_cuda import VIEW_STRIDE

__all__ = [
    "sf_forward_project_3d_torch", "sf_back_project_3d_torch",
    "sf_precompute_footprints_3d_torch",
    "sf_forward_project_3d_csr_torch", "sf_back_project_3d_csr_torch",
]

_CHUNK = 1 << 21          # voxels per chunk (float64 temporaries ~ 1 GB at 2M voxels x ~60 doubles)


# --------------------------------------------------------------------------- footprint pieces
def _trap_cdf(x, t0, t1, t2, t3):
    """Integral of the unit-height trapezoid (t0,t1,t2,t3) from -inf to x."""
    A1 = 0.5 * (t1 - t0); A2 = t2 - t1; A3 = 0.5 * (t3 - t2)
    d01 = (t1 - t0).clamp_min(1e-30); d23 = (t3 - t2).clamp_min(1e-30)
    y1 = 0.5 * (x - t0) ** 2 / d01
    y3 = A1 + A2 + (x - t2) - 0.5 * (x - t2) ** 2 / d23
    return torch.where(x <= t0, torch.zeros_like(x),
           torch.where(x < t1, y1,
           torch.where(x < t2, A1 + (x - t1),
           torch.where(x < t3, y3, A1 + A2 + A3))))


def _view_unpack(view):
    v = view.to(torch.float64)
    return (v[0:3], v[3:6], v[6:9], v[9:12], v[12:15],
            float(v[15]), float(v[16]), int(round(float(v[17]))), int(round(float(v[18]))))


def _project(P, S, C, U, V, N, du, dv, nu, nv):
    """Central projection of points P [n,3] on the panel -> continuous pixel coords (u, v), magnification t."""
    r = P - S
    denom = r @ N
    ok = denom.abs() > 1e-12
    t = ((C - S) @ N) / torch.where(ok, denom, torch.ones_like(denom))
    ok = ok & (t > 0)
    H = S + t[:, None] * r - C
    up = (H @ U) / du + 0.5 * (nu - 1) + 0.5
    vp = (H @ V) / dv + 0.5 * (nv - 1) + 0.5
    return up, vp, t, ok


def _voxel_centres(idx, n_y, n_z, M, b):
    i = idx // (n_y * n_z); j = (idx // n_z) % n_y; k = idx % n_z
    return torch.stack([i, j, k], 1).to(M.dtype) @ M.T + b


def _footprints_view(cen, M, view, valid_v):
    """Footprint of voxel centres ``cen`` [n,3] on one view.

    Returns ``sel`` (voxel rows with a footprint), ``pix`` [m] flat in-view pixel index
    (iu*n_v+iv) and ``w`` [m] weights, plus ``vox`` [m] the row of each entry.
    """
    S, C, U, V, N, du, dv, nu, nv = _view_unpack(view)
    ex, ey, ez = M[:, 0] / 2, M[:, 1] / 2, M[:, 2] / 2
    lx, ly, lz = 2 * ex.norm(), 2 * ey.norm(), 2 * ez.norm()
    hd = 0.5 * math.sqrt(float(lx * lx + ly * ly + lz * lz))
    up, vp, t, ok = _project(cen, S, C, U, V, N, du, dv, nu, nv)
    mu = hd * t / du + 1; mv = hd * t / dv + 1
    ok = ok & (up >= -mu) & (up <= nu + mu) & (vp >= -mv) & (vp <= nv + mv)
    sel = torch.nonzero(ok).squeeze(1)
    empty = (sel, sel.new_empty(0), cen.new_empty(0), sel.new_empty(0))
    if sel.numel() == 0:
        return empty
    c = cen[sel]
    # amplitude: chord through the voxel centre along the dominant axis
    d = c - S; d = d / d.norm(dim=1, keepdim=True)
    amp = torch.full((c.shape[0],), 1e30, dtype=c.dtype, device=c.device)
    for e, l in ((ex, lx), (ey, ly), (ez, lz)):
        a = (d @ (e / e.norm())).abs()
        amp = torch.where(a > 1e-9, torch.minimum(amp, l / a), amp)
    # transaxial trapezoid from the 4 (x,y) corners at the voxel k-centre, axial rectangle from the 2 k-faces
    us = torch.stack([_project(c + s1 * ex + s2 * ey, S, C, U, V, N, du, dv, nu, nv)[0]
                      for s1, s2 in ((1, 1), (1, -1), (-1, 1), (-1, -1))], 1).sort(1).values
    t0, t1, t2, t3 = us.unbind(1)
    v0 = _project(c + ez, S, C, U, V, N, du, dv, nu, nv)[1]
    v1 = _project(c - ez, S, C, U, V, N, du, dv, nu, nv)[1]
    r0, r1 = torch.minimum(v0, v1), torch.maximum(v0, v1)
    iu0 = t0.floor().clamp(0, nu - 1).long(); iu1 = (t3.ceil() - 1).clamp(0, nu - 1).long()
    iv0 = r0.floor().clamp(0, nv - 1).long(); iv1 = (r1.ceil() - 1).clamp(0, nv - 1).long()
    nu_span = int((iu1 - iu0).max()) + 1; nv_span = int((iv1 - iv0).max()) + 1
    du_ = torch.arange(nu_span, device=c.device); dv_ = torch.arange(nv_span, device=c.device)
    iu = iu0[:, None] + du_[None, :]                                                        # [n, nu_span]
    iv = iv0[:, None] + dv_[None, :]                                                        # [n, nv_span]
    wu = _trap_cdf(iu.to(c.dtype) + 1, t0[:, None], t1[:, None], t2[:, None], t3[:, None]) \
       - _trap_cdf(iu.to(c.dtype), t0[:, None], t1[:, None], t2[:, None], t3[:, None])
    wv = (torch.minimum(iv.to(c.dtype) + 1, r1[:, None]) - torch.maximum(iv.to(c.dtype), r0[:, None])).clamp_min(0)
    wu = wu * (iu <= iu1[:, None]) * (iu < nu); wv = wv * (iv <= iv1[:, None]) * (iv < nv)
    w = amp[:, None, None] * wu[:, :, None] * wv[:, None, :]                                # [n, nu_span, nv_span]
    pix = iu.clamp_max(nu - 1)[:, :, None] * nv + iv.clamp_max(nv - 1)[:, None, :]
    if valid_v is not None:
        w = w * valid_v.reshape(-1)[pix].to(w.dtype)
    nz_ = w != 0
    vox = sel[:, None, None].expand_as(w)[nz_]
    return sel, pix[nz_], w[nz_], vox


def _iter_views(views, view_off, valid):
    for wv in range(views.shape[0]):
        o0, o1 = int(view_off[wv]), int(view_off[wv + 1])
        vmask = None if valid.numel() == 0 else valid[o0:o1]
        yield wv, o0, o1, vmask


# --------------------------------------------------------------------------- on-the-fly
def sf_forward_project_3d_torch(volume, M, b, views, view_off, valid):
    """volume [X,Y,Z] or [B,X,Y,Z] -> flat sinogram [n_pix] or [B, n_pix]."""
    batched = volume.dim() == 4
    vol = volume if batched else volume[None]
    Bn, n_x, n_y, n_z = vol.shape
    N = n_x * n_y * n_z
    n_pix = int(view_off[-1])
    dev = vol.device
    M64 = M.to(dev, torch.float64); b64 = b.to(dev, torch.float64)
    vf = vol.reshape(Bn, N).to(torch.float64)
    out = torch.zeros(Bn, n_pix, dtype=torch.float64, device=dev)
    for c0 in range(0, N, _CHUNK):
        idx = torch.arange(c0, min(N, c0 + _CHUNK), device=dev)
        cen = _voxel_centres(idx, n_y, n_z, M64, b64)
        for wv, o0, o1, vmask in _iter_views(views, view_off, valid):
            sel, pix, w, vox = _footprints_view(cen, M64, views[wv], vmask)
            if pix.numel() == 0:
                continue
            out.index_add_(1, o0 + pix, vf[:, c0 + vox] * w[None, :])
    out = out.to(volume.dtype)
    return out if batched else out[0]


def sf_back_project_3d_torch(sino, M, b, views, view_off, valid, n_x, n_y, n_z):
    """flat sinogram [n_pix] or [B, n_pix] -> volume [X,Y,Z] or [B,X,Y,Z]."""
    batched = sino.dim() == 2
    s = sino if batched else sino[None]
    Bn = s.shape[0]; N = n_x * n_y * n_z; dev = s.device
    M64 = M.to(dev, torch.float64); b64 = b.to(dev, torch.float64)
    s64 = s.to(torch.float64)
    out = torch.zeros(Bn, N, dtype=torch.float64, device=dev)
    for c0 in range(0, N, _CHUNK):
        idx = torch.arange(c0, min(N, c0 + _CHUNK), device=dev)
        cen = _voxel_centres(idx, n_y, n_z, M64, b64)
        for wv, o0, o1, vmask in _iter_views(views, view_off, valid):
            sel, pix, w, vox = _footprints_view(cen, M64, views[wv], vmask)
            if pix.numel() == 0:
                continue
            out.index_add_(1, c0 + vox, s64[:, o0 + pix] * w[None, :])
    out = out.to(sino.dtype).reshape(Bn, n_x, n_y, n_z)
    return out if batched else out[0]


# --------------------------------------------------------------------------- CSR
def sf_precompute_footprints_3d_torch(n_x, n_y, n_z, M, b, views, view_off, valid):
    """Per-voxel CSR lists (ptr int64 [N+1], idx int32 [nnz], w float32 [nnz]) — same layout as the CUDA version."""
    N = n_x * n_y * n_z; dev = views.device
    M64 = M.to(dev, torch.float64); b64 = b.to(dev, torch.float64)
    vox_l, pix_l, w_l = [], [], []
    for c0 in range(0, N, _CHUNK):
        idx = torch.arange(c0, min(N, c0 + _CHUNK), device=dev)
        cen = _voxel_centres(idx, n_y, n_z, M64, b64)
        for wv, o0, o1, vmask in _iter_views(views, view_off, valid):
            sel, pix, w, vox = _footprints_view(cen, M64, views[wv], vmask)
            if pix.numel() == 0:
                continue
            vox_l.append(c0 + vox); pix_l.append(o0 + pix); w_l.append(w)
    if vox_l:
        vox = torch.cat(vox_l); pix = torch.cat(pix_l); w = torch.cat(w_l)
        order = torch.argsort(vox, stable=True)
        vox, pix, w = vox[order], pix[order], w[order]
    else:
        vox = torch.empty(0, dtype=torch.int64, device=dev); pix = vox.clone(); w = torch.empty(0, dtype=torch.float64, device=dev)
    counts = torch.bincount(vox, minlength=N)
    ptr = torch.zeros(N + 1, dtype=torch.int64, device=dev); ptr[1:] = torch.cumsum(counts, 0)
    return ptr, pix.to(torch.int32), w.to(torch.float32)


def _csr_rows(ptr):
    counts = ptr[1:] - ptr[:-1]
    return torch.repeat_interleave(torch.arange(counts.numel(), device=ptr.device), counts)


def sf_forward_project_3d_csr_torch(volume, ptr, idx, w, n_pix):
    batched = volume.dim() == 4
    vol = volume if batched else volume[None]
    Bn = vol.shape[0]; vf = vol.reshape(Bn, -1)
    rows = _csr_rows(ptr)
    out = torch.zeros(Bn, int(n_pix), dtype=vol.dtype, device=vol.device)
    out.index_add_(1, idx.long(), vf[:, rows] * w.to(vol.dtype)[None, :])
    return out if batched else out[0]


def sf_back_project_3d_csr_torch(sino, ptr, idx, w, n_x, n_y, n_z):
    batched = sino.dim() == 2
    s = sino if batched else sino[None]
    Bn = s.shape[0]; N = n_x * n_y * n_z
    rows = _csr_rows(ptr)
    out = torch.zeros(Bn, N, dtype=s.dtype, device=s.device)
    out.index_add_(1, rows, s[:, idx.long()] * w.to(s.dtype)[None, :])
    out = out.reshape(Bn, n_x, n_y, n_z)
    return out if batched else out[0]
