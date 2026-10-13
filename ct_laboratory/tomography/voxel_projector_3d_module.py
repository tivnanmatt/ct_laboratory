# File: voxel_projector_3d_module.py
"""Voxel-driven 3-D projector using SEPARABLE FOOTPRINTS (SF-TR).

Reference: Y. Long, J. A. Fessler, J. M. Balter, "3D forward and back-projection
for X-ray CT using separable footprints", IEEE TMI 29(11):1839-1850, 2010.

Whereas :class:`RayProjector3D` (= ``CTProjector3DModule``) stores, for every
RAY, the parameters of its voxel intersections (``tvals``), this class stores
(or recomputes), for every VOXEL, the list of detector pixels its footprint
covers and the corresponding weights.  Consequences:

* back projection is a pure gather -> no atomics, deterministic;
* forward projection is a scatter with atomics on the (small) sinogram;
* the weights integrate the footprint over the whole pixel FACE, i.e. this is a
  finite-pixel ("area") detector model, not a centre-ray model.

Detector model
--------------
The detector is a list of *views*; a view is one (source, flat panel) pair:
source position, panel centre, unit column vector ``u`` (transaxial), unit row
vector ``v`` (axial), pitches ``du, dv`` and pixel counts ``n_u, n_v``.  Pixel
``(iu, iv)`` is centred at ``centre + (iu-(n_u-1)/2)·du·u + (iv-(n_v-1)/2)·dv·v``.
This matches the static-CT module description in ``staticct_projector_3d`` where
``module_orientations[:, :, 0] = u`` (side), ``[:, :, 1] = v`` (up).

Sinogram ordering (identical to ct_laboratory's ray order: column outer, row
inner): flat index ``= view_off[view] + iu * n_v + iv``.  :meth:`rays` returns
``src, dst`` in exactly this order, so a :class:`RayProjector3D` built from
them produces a sinogram that is element-for-element comparable.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch

from .projector_3d_base import Projector3D
from .voxel_projector_3d_cuda import (
    VIEW_STRIDE,
    sf_forward_project_3d_cuda,
    sf_back_project_3d_cuda,
    sf_precompute_footprints_3d_cuda,
    sf_forward_project_3d_csr_cuda,
    sf_back_project_3d_csr_cuda,
    sf_precompute_columns_3d_cuda,
    sf_back_project_3d_col_cuda,
    sf_forward_project_3d_col_fused_cuda,
    sf_back_project_3d_col_fused_cuda,
    SF_MAXB,
)
from .voxel_projector_3d_torch import (
    sf_back_project_3d_torch,
    sf_precompute_footprints_3d_torch,
    sf_back_project_3d_csr_torch,
)
from .voxel_projector_3d_autograd import VoxelProjector3DFunction

__all__ = ["VoxelProjector3D", "make_views", "views_from_module_orientations"]


# ---------------------------------------------------------------------------
# view construction helpers
# ---------------------------------------------------------------------------
def _unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def make_views(
    source_positions: torch.Tensor,   # [n_src, 3]
    panel_centers: torch.Tensor,      # [n_pan, 3]
    panel_u: torch.Tensor,            # [n_pan, 3]  column direction
    panel_v: torch.Tensor,            # [n_pan, 3]  row direction
    pitch_u, pitch_v,                 # float or [n_pan]
    n_u, n_v,                         # int   or [n_pan]
    pairs: Optional[torch.Tensor] = None,   # [n_view, 2] (source, panel); None = all pairs
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the ``[n_view, 19]`` float32 view table and the ``[n_view, 2]`` pair list."""
    if device is None:
        device = source_positions.device
    S = source_positions.to(device=device, dtype=torch.float32)
    C = panel_centers.to(device=device, dtype=torch.float32)
    U = _unit(panel_u.to(device=device, dtype=torch.float32))
    V = panel_v.to(device=device, dtype=torch.float32)
    V = _unit(V - (V * U).sum(-1, keepdim=True) * U)      # orthogonalise row w.r.t. column
    N = _unit(torch.linalg.cross(U, V))
    n_pan = C.shape[0]
    pu = torch.as_tensor(pitch_u, dtype=torch.float32, device=device).expand(n_pan) if torch.as_tensor(pitch_u).numel() == 1 else torch.as_tensor(pitch_u, dtype=torch.float32, device=device)
    pv = torch.as_tensor(pitch_v, dtype=torch.float32, device=device).expand(n_pan) if torch.as_tensor(pitch_v).numel() == 1 else torch.as_tensor(pitch_v, dtype=torch.float32, device=device)
    nu = torch.as_tensor(n_u, dtype=torch.float32, device=device).expand(n_pan) if torch.as_tensor(n_u).numel() == 1 else torch.as_tensor(n_u, dtype=torch.float32, device=device)
    nv = torch.as_tensor(n_v, dtype=torch.float32, device=device).expand(n_pan) if torch.as_tensor(n_v).numel() == 1 else torch.as_tensor(n_v, dtype=torch.float32, device=device)

    if pairs is None:
        s_idx = torch.arange(S.shape[0], device=device).repeat_interleave(n_pan)
        p_idx = torch.arange(n_pan, device=device).repeat(S.shape[0])
        pairs = torch.stack([s_idx, p_idx], dim=1)
    pairs = pairs.to(device=device, dtype=torch.int64)
    s_idx, p_idx = pairs[:, 0], pairs[:, 1]

    views = torch.cat([
        S[s_idx], C[p_idx], U[p_idx], V[p_idx], N[p_idx],
        pu[p_idx, None], pv[p_idx, None], nu[p_idx, None], nv[p_idx, None],
    ], dim=1).contiguous()
    assert views.shape[1] == VIEW_STRIDE
    return views, pairs


def views_from_module_orientations(
    source_positions: torch.Tensor,     # [n_src, 3]
    module_centers: torch.Tensor,       # [n_mod, 3]
    module_orientations: torch.Tensor,  # [n_mod, 3, 3], columns = [side(u), up(v), normal]
    pitch_u, pitch_v, n_u, n_v,
    source_module_mask: Optional[torch.Tensor] = None,   # [n_src, n_mod] bool
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Views from the static-CT module description used throughout ct_laboratory."""
    pairs = None
    if source_module_mask is not None:
        pairs = torch.nonzero(source_module_mask, as_tuple=False)
    return make_views(source_positions, module_centers,
                      module_orientations[:, :, 0], module_orientations[:, :, 1],
                      pitch_u, pitch_v, n_u, n_v, pairs=pairs, device=device)


# ---------------------------------------------------------------------------
# the projector
# ---------------------------------------------------------------------------
class VoxelProjector3D(Projector3D):
    """Voxel-driven separable-footprint projector (SF-TR).

    Parameters
    ----------
    n_x, n_y, n_z : volume shape (axis k = third column of M is the axial direction)
    M, b          : ``world = M @ (i,j,k) + b`` (3x3, 3)
    views         : ``[n_view, 19]`` table from :func:`make_views`
    valid         : optional pixel mask, ``[n_view, n_u, n_v]`` (uniform panels) or flat ``[n_pix]``;
                    masked pixels are never written or read
    precompute_footprints : build the per-voxel CSR (pixel index, weight) lists at init
    backend       : ``'cuda'`` (custom kernels, CUDA device only) or ``'torch'`` (pure-PyTorch
                    implementation of the same footprints, runs on any device incl. CPU) —
                    the same two options as :class:`RayProjector3D`
    cache         : ``'none'`` (on the fly), ``'column'`` (per transaxial column: culled view list
                    + transaxial trapezoid stored at both end slices, 36 B per (column, view) entry, CUDA only;
                    the axial rectangle and amplitude stay exact per voxel) or ``'csr'`` (full
                    per-voxel pixel/weight lists, 8 B per nonzero).  ``precompute_footprints=True``
                    is the same as ``cache='csr'``.

    Which cache to use: 'csr' is the fastest when it fits (few views); 'column' is the
    production choice for large multi-source geometries where the CSR would be tens of GB;
    'none' needs no memory but pays the view-rejection cost on every call.
    """

    def __init__(self, n_x, n_y, n_z, M, b, views, valid=None,
                 device=None, precompute_footprints=False, backend="cuda", cache="none"):
        super().__init__()
        if backend not in ("cuda", "torch"):
            raise ValueError(f"backend must be 'cuda' or 'torch', got {backend!r}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        if backend == "cuda" and device.type != "cuda":
            raise ValueError("backend='cuda' needs a CUDA device; use backend='torch' on the CPU")
        self.backend = backend
        self.n_x, self.n_y, self.n_z = int(n_x), int(n_y), int(n_z)

        if M.dim() > 2:
            M = M[0]
        if b.dim() > 1:
            b = b[0]
        M = M.to(dtype=torch.float32).reshape(3, 3).contiguous()
        b = b.to(dtype=torch.float32).reshape(3).contiguous()
        views = views.to(dtype=torch.float32).contiguous()
        assert views.dim() == 2 and views.shape[1] == VIEW_STRIDE, "views must be [n_view, 19]"

        n_u = views[:, 17].round().to(torch.int64)
        n_v = views[:, 18].round().to(torch.int64)
        npix_view = n_u * n_v
        view_off = torch.zeros(views.shape[0] + 1, dtype=torch.int64)
        view_off[1:] = torch.cumsum(npix_view.cpu(), 0)
        self._n_pix = int(view_off[-1])
        self._uniform = bool((n_u == n_u[0]).all() and (n_v == n_v[0]).all())
        self._n_u0, self._n_v0 = int(n_u[0]), int(n_v[0])

        if valid is None:
            valid_t = torch.empty(0, dtype=torch.uint8)
        else:
            valid_t = valid.to(torch.uint8).reshape(-1).contiguous()
            assert valid_t.numel() == self._n_pix, f"valid must have {self._n_pix} entries"

        self.register_buffer("M", M)
        self.register_buffer("b", b)
        self.register_buffer("views", views)
        self.register_buffer("view_off", view_off)
        self.register_buffer("valid", valid_t)
        self.register_buffer("fp_ptr", None)
        self.register_buffer("fp_idx", None)
        self.register_buffer("fp_w", None)
        self.register_buffer("col_ptr", None)
        self.register_buffer("col_view", None)
        self.register_buffer("col_trap", None)
        self.to(device)
        if precompute_footprints or cache == "csr":
            self.precompute_footprints()
        elif cache == "column":
            self.precompute_columns()
        elif cache != "none":
            raise ValueError(f"cache must be 'none', 'column' or 'csr', got {cache!r}")

    # ------------------------------------------------------------ properties
    @property
    def kind(self) -> str:
        return "voxel"

    @property
    def n_view(self) -> int:
        return int(self.views.shape[0])

    @property
    def n_pix(self) -> int:
        return self._n_pix

    @property
    def n_ray(self) -> int:
        return self._n_pix

    @property
    def sino_shape(self) -> tuple[int, ...]:
        """``(n_view, n_u, n_v)`` when all panels have the same size, else ``(n_pix,)``."""
        return (self.n_view, self._n_u0, self._n_v0) if self._uniform else (self._n_pix,)

    @property
    def has_footprints(self) -> bool:
        return self.fp_ptr is not None

    @property
    def has_columns(self) -> bool:
        return self.col_ptr is not None

    @property
    def column_entries(self) -> int:
        return int(self.col_view.numel()) if self.has_columns else 0

    @property
    def column_bytes(self) -> int:
        if not self.has_columns:
            return 0
        return self.col_ptr.numel() * 8 + self.col_view.numel() * 4 + self.col_trap.numel() * 4

    @property
    def cache(self) -> str:
        return "csr" if self.has_footprints else ("column" if self.has_columns else "none")

    @property
    def nnz(self) -> int:
        return int(self.fp_idx.numel()) if self.has_footprints else 0

    @property
    def footprint_bytes(self) -> int:
        if not self.has_footprints:
            return 0
        return self.fp_ptr.numel() * 8 + self.fp_idx.numel() * 4 + self.fp_w.numel() * 4

    # --------------------------------------------------------------- rays
    def rays(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``src, dst`` ``[n_pix, 3]`` in flat sinogram order (view, column, row).

        Feed these to :class:`RayProjector3D` to get a ray-driven projector whose
        sinogram is element-for-element comparable with this one.
        """
        v = self.views
        src_l, dst_l = [], []
        for w in range(self.n_view):
            r = v[w]
            S, C, U, V = r[0:3], r[3:6], r[6:9], r[9:12]
            du, dv, nu, nv = float(r[15]), float(r[16]), int(round(float(r[17]))), int(round(float(r[18])))
            iu = torch.arange(nu, device=v.device, dtype=torch.float32) - 0.5 * (nu - 1)
            iv = torch.arange(nv, device=v.device, dtype=torch.float32) - 0.5 * (nv - 1)
            IU, IV = torch.meshgrid(iu, iv, indexing="ij")           # column outer, row inner
            dst = C[None, :] + (IU.reshape(-1, 1) * du) * U[None, :] + (IV.reshape(-1, 1) * dv) * V[None, :]
            dst_l.append(dst)
            src_l.append(S[None, :].expand(dst.shape[0], 3))
        return torch.cat(src_l, 0).contiguous(), torch.cat(dst_l, 0).contiguous()

    def clone_to(self, device) -> "VoxelProjector3D":
        """A copy of this projector (grid, views, valid mask and the precomputed caches) on another device, built under that device."""
        device = torch.device(device)
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            valid = self.valid if self.valid.numel() else None
            op = VoxelProjector3D(self.n_x, self.n_y, self.n_z, self.M, self.b, self.views, valid=valid, device=device, backend=self.backend, cache="none")
            for k in ("fp_ptr", "fp_idx", "fp_w", "col_ptr", "col_view", "col_trap"):
                t = getattr(self, k)
                if t is not None: setattr(op, k, t.to(device))
        op.meta = getattr(self, "meta", {}); return op

    # ------------------------------------------------------------- save / load
    def save(self, path: str, **meta) -> None:
        """Write the whole operator (grid, views, valid mask and the precomputed cache) to one .pt file; meta is stored alongside.
        VoxelProjector3D.load rebuilds it without recomputing the cache."""
        torch.save(dict(format="ct_laboratory.VoxelProjector3D/1", n_x=self.n_x, n_y=self.n_y, n_z=self.n_z, backend=self.backend,
                        buffers={k: (None if v is None else v.detach().cpu()) for k, v in self._buffers.items()}, meta=meta), path)

    @classmethod
    def load(cls, path: str, device=None) -> "VoxelProjector3D":
        """Rebuild a projector written by save() on device (default: the first CUDA device, else the CPU)."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        assert d.get("format", "").startswith("ct_laboratory.VoxelProjector3D/"), f"{path} is not a saved VoxelProjector3D"
        b = d["buffers"]; valid = b["valid"] if b["valid"] is not None and b["valid"].numel() else None
        op = cls(d["n_x"], d["n_y"], d["n_z"], b["M"], b["b"], b["views"], valid=valid, device=device, backend=d["backend"], cache="none")
        for k in ("fp_ptr", "fp_idx", "fp_w", "col_ptr", "col_view", "col_trap"):
            if b.get(k) is not None: setattr(op, k, b[k].to(op.M.device))
        op.meta = d.get("meta", {}); return op

    # ---------------------------------------------------------- footprints
    def precompute_footprints(self) -> int:
        """Build per-voxel CSR (pixel index, weight) lists. Returns nnz."""
        fn = sf_precompute_footprints_3d_cuda if self.backend == "cuda" else sf_precompute_footprints_3d_torch
        ptr, idx, w = fn(self.n_x, self.n_y, self.n_z, self.M, self.b, self.views, self.view_off, self.valid)
        self.fp_ptr, self.fp_idx, self.fp_w = ptr, idx, w
        return self.nnz

    def clear_footprints(self) -> None:
        self.fp_ptr = self.fp_idx = self.fp_w = None

    def precompute_columns(self) -> int:
        """Build the per-column culled view lists with stored trapezoids (CUDA backend). Returns #entries."""
        if self.backend != "cuda":
            raise NotImplementedError("the column cache is implemented for backend='cuda' only")
        ptr, cv, ct = sf_precompute_columns_3d_cuda(
            self.n_x, self.n_y, self.n_z, self.M, self.b, self.views, self.view_off, self.valid)
        self.col_ptr, self.col_view, self.col_trap = ptr, cv, ct
        return self.column_entries

    def clear_columns(self) -> None:
        self.col_ptr = self.col_view = self.col_trap = None

    # ----------------------------------------------------------- projection
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Autograd-aware forward projection."""
        return VoxelProjector3DFunction.apply(
            volume.contiguous(), self.M, self.b, self.views, self.view_off, self.valid,
            self.fp_ptr, self.fp_idx, self.fp_w, self._n_pix, self.backend,
            self.col_ptr, self.col_view, self.col_trap)

    def forward_project(self, volume: torch.Tensor) -> torch.Tensor:
        """A batch ``[B, X, Y, Z]`` goes through the fused kernel when the column cache is available (one footprint per voxel for the
        whole batch; no autograd graph); everything else is :meth:`forward`."""
        if volume.dim() == 4 and volume.shape[0] > 1 and self.fused_available:
            return self.forward_project_batched(volume)
        return self.forward(volume)

    # ------------------------------------------------- fused batched / rolled projection (column cache, CUDA backend)
    @property
    def fused_available(self) -> bool:
        return self.backend == "cuda" and self.has_columns

    def _fused_args(self):
        return (self.M, self.b, self.views, self.view_off, self.valid, self.col_ptr, self.col_view, self.col_trap)

    forward_replicas = 1   # sinogram copies for the fused forward scatter, summed afterwards
    forward_replica_mode = 0   # 0: replica = slice index mod replicas; 1: depth along the ray; 2: slice + depth

    def forward_project_batched(self, volumes: torch.Tensor) -> torch.Tensor:
        """``[B, X, Y, Z] -> [B, n_pix]``: the same projector applied to every volume, with each voxel's footprint computed once for the
        whole batch (chunks of SF_MAXB).  Falls back to one projection per volume without the column cache."""
        volumes = volumes.contiguous()
        if not self.fused_available:
            return torch.stack([self.forward(v) for v in volumes])
        out = torch.empty((volumes.shape[0], self.n_pix), device=volumes.device, dtype=torch.float32)
        for b0 in range(0, volumes.shape[0], SF_MAXB):
            out[b0:b0 + SF_MAXB] = sf_forward_project_3d_col_fused_cuda(volumes[b0:b0 + SF_MAXB], *self._fused_args(), replicas=self.forward_replicas, replica_mode=self.forward_replica_mode, n_pix=self.n_pix)
        return out

    def back_project_batched(self, sinograms: torch.Tensor) -> torch.Tensor:
        """``[B, n_pix] -> [B, X, Y, Z]``, the adjoint of :meth:`forward_project_batched`."""
        sinograms = sinograms.contiguous()
        if not self.fused_available:
            return torch.stack([self.back_project(y) for y in sinograms])
        out = torch.empty((sinograms.shape[0], self.n_x, self.n_y, self.n_z), device=sinograms.device, dtype=torch.float32)
        for b0 in range(0, sinograms.shape[0], SF_MAXB):
            out[b0:b0 + SF_MAXB] = sf_back_project_3d_col_fused_cuda(sinograms[b0:b0 + SF_MAXB], *self._fused_args(), self.n_x, self.n_y, self.n_z, n_pix=self.n_pix)
        return out

    def forward_project_rolled(self, volume: torch.Tensor, batch: int, z0: int = 0, zstep: int = 1) -> torch.Tensor:
        """Multi-rotation hook.  ``volume [X, Y, Z_tot]``; element ``i`` of the batch is this projector applied to the window of
        slices ``[z0 + i*zstep, z0 + i*zstep + Z)``.  Slices outside the volume are zero (padding; never wrapped).  Returns
        ``[batch, n_pix]``.  One footprint per window voxel serves the whole batch (chunks of SF_MAXB)."""
        volume = volume.contiguous()
        assert volume.dim() == 3 and tuple(volume.shape[:2]) == (self.n_x, self.n_y), "volume must be [X, Y, Z_tot]"
        if not self.fused_available:                                         # same semantics, one window at a time
            nt = volume.shape[2]; outs = []
            for i in range(batch):
                a0 = z0 + i * zstep; win = torch.zeros((self.n_x, self.n_y, self.n_z), device=volume.device, dtype=torch.float32)
                lo, hi = max(a0, 0), min(a0 + self.n_z, nt)
                if hi > lo: win[..., lo - a0:hi - a0] = volume[..., lo:hi]
                outs.append(self.forward(win))
            return torch.stack(outs)
        out = torch.empty((batch, self.n_pix), device=volume.device, dtype=torch.float32)
        for b0 in range(0, batch, SF_MAXB):
            nb = min(SF_MAXB, batch - b0)
            out[b0:b0 + nb] = sf_forward_project_3d_col_fused_cuda(volume, *self._fused_args(), batch=nb, nz_win=self.n_z, z0=z0 + b0 * zstep, zstep=zstep, replicas=self.forward_replicas, replica_mode=self.forward_replica_mode, n_pix=self.n_pix)
        return out

    def back_project_rolled(self, sinograms: torch.Tensor, out: torch.Tensor, z0: int = 0, zstep: int = 1) -> torch.Tensor:
        """Adjoint of :meth:`forward_project_rolled`: the back projection of element ``i`` is ADDED to slices
        ``[z0 + i*zstep, + Z)`` of ``out [X, Y, Z_tot]`` (slices outside ``out`` are dropped, never wrapped).  Returns ``out``."""
        sinograms = sinograms.contiguous()
        assert out.dim() == 3 and tuple(out.shape[:2]) == (self.n_x, self.n_y) and out.is_contiguous(), "out must be a contiguous [X, Y, Z_tot]"
        if not self.fused_available:
            nt = out.shape[2]
            for i in range(sinograms.shape[0]):
                a0 = z0 + i * zstep; lo, hi = max(a0, 0), min(a0 + self.n_z, nt)
                if hi > lo: out[..., lo:hi] += self.back_project(sinograms[i]).reshape(self.n_x, self.n_y, self.n_z)[..., lo - a0:hi - a0]
            return out
        for b0 in range(0, sinograms.shape[0], SF_MAXB):
            nb = min(SF_MAXB, sinograms.shape[0] - b0)
            sf_back_project_3d_col_fused_cuda(sinograms[b0:b0 + nb], *self._fused_args(), self.n_x, self.n_y, self.n_z, out=out, z0=z0 + b0 * zstep, zstep=zstep, n_pix=self.n_pix)
        return out

    def back_project(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Exact adjoint of :meth:`forward_project` (no autograd graph).  A batch ``[B, n_pix]`` goes through the fused kernel when the
        column cache is available (one footprint per voxel for the whole batch)."""
        sinogram = sinogram.contiguous()
        if sinogram.dim() == 2 and sinogram.shape[0] > 1 and self.fused_available:
            return self.back_project_batched(sinogram)
        cuda = self.backend == "cuda"
        if self.has_footprints:
            fn = sf_back_project_3d_csr_cuda if cuda else sf_back_project_3d_csr_torch
            return fn(sinogram, self.fp_ptr, self.fp_idx, self.fp_w, self.n_x, self.n_y, self.n_z)
        if self.has_columns:
            return sf_back_project_3d_col_cuda(sinogram, self.M, self.b, self.views, self.view_off, self.valid,
                                               self.col_ptr, self.col_view, self.col_trap, self.n_x, self.n_y, self.n_z)
        fn = sf_back_project_3d_cuda if cuda else sf_back_project_3d_torch
        return fn(sinogram, self.M, self.b, self.views, self.view_off, self.valid, self.n_x, self.n_y, self.n_z)

    # ------------------------------------------------------------- utility
    def sinogram_view(self, sino: torch.Tensor) -> torch.Tensor:
        """Reshape a flat sinogram to ``[..., n_view, n_u, n_v]`` (uniform panels only)."""
        if not self._uniform:
            raise ValueError("panels are not all the same size")
        return sino.reshape(*sino.shape[:-1], self.n_view, self._n_u0, self._n_v0)

    def extra_repr(self) -> str:
        if self.has_footprints:
            fp = f", cache=csr nnz={self.nnz} ({self.footprint_bytes/1e6:.1f} MB)"
        elif self.has_columns:
            fp = f", cache=column entries={self.column_entries} ({self.column_bytes/1e6:.1f} MB)"
        else:
            fp = ", cache=none (on-the-fly)"
        return f"kind=voxel(SF-TR), backend={self.backend}, volume={self.volume_shape}, n_view={self.n_view}, n_pix={self.n_pix}{fp}"
