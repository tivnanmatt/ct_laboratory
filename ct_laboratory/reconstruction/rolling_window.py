"""Multi-rotation step-and-shoot reconstruction with a rolling active window.

Geometry: one rotation = a fixed set of (source, panel) *views*; rotation j is the same set
shifted by ``j * dz_rot`` along z.  The volume has ``n_tot = n_win + n_rot - 1`` slices of
height ``dz_slice``; rotation j illuminates the window ``[j, j + n_win)``.  One
:class:`VoxelProjector3D` over the window serves every rotation (``fwd``/``adj`` roll it),
split across all available GPUs by view range.

Data: ``y`` ``[n_rot, n_ray]`` post-log line integrals in *model* rotation order and
``w`` ``[n_rot, n_ray]`` statistical weights / masks.  The MAP problem
``min 0.5 ||A x - y||_w^2 + 0.5 beta ||D x||^2`` is solved by PCG with the k-term window
eigen-preconditioner, warm-started level to level in a multi-resolution cascade.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..sparse_eigen_preconditioner import SparseEigenDecomposition
from ..tomography import VoxelProjector3D, split_voxel_projector
from ..tomography.voxel_projector_3d_module import make_views
from ..workflow.gpus import available_devices

__all__ = ["StepAndShootGeometry", "ProjectorSpec", "decimate", "RollingWindowOperator", "bin_sinogram", "window_eigen",
           "lambda_max", "quadratic_penalty_grad", "pcg", "upsample_inplane", "cascade"]


# ------------------------------------------------------------------ geometry
@dataclass
class StepAndShootGeometry:
    """Views of ONE rotation at full detector resolution, plus the stepping.

    S, C, U, V : [n_view, 3] source position, panel centre, column (u) and row (v) unit vectors
    pitch_u, pitch_v : pixel pitch (mm) at bin 1;  n_u, n_v : pixels per panel at bin 1
    n_rot, dz_rot : number of rotations and the z step between consecutive MODEL rotations (mm)
    z_mid : beam mid-plane z of rotation 0 (mm), for volume placement
    """
    S: torch.Tensor; C: torch.Tensor; U: torch.Tensor; V: torch.Tensor
    pitch_u: float; pitch_v: float; n_u: int; n_v: int
    n_rot: int; dz_rot: float; z_mid: float
    meta: dict = field(default_factory=dict)

    @property
    def n_view(self) -> int:
        return int(self.S.shape[0])

    def views(self, B: int, device) -> torch.Tensor:
        idx = torch.arange(self.n_view)
        v, _ = make_views(self.S, self.C, self.U, self.V, self.pitch_u * B, self.pitch_v * B,
                          self.n_u // B, self.n_v // B, pairs=torch.stack([idx, idx], 1), device=device)
        return v

    def ray_z_span(self, B: int, fov_radius: float) -> tuple[float, float]:
        """z range of all rays (panel corners) inside the transaxial FOV."""
        hu = self.pitch_u * B * (self.n_u // B) / 2
        hv = self.pitch_v * B * (self.n_v // B) / 2
        S, C, U, V = self.S, self.C, self.U, self.V
        Vn = V - (V * U).sum(-1, keepdim=True) * U
        Vn = Vn / Vn.norm(dim=-1, keepdim=True)
        t = torch.linspace(0, 1, 4001)
        zmin, zmax = float("inf"), -float("inf")
        for su in (-1, 1):
            for sv in (-1, 1):
                D = C + su * hu * U + sv * hv * Vn
                P = S[:, None, :] + t[None, :, None] * (D - S)[:, None, :]
                inside = P[..., :2].norm(dim=-1) <= fov_radius
                z = P[..., 2]
                zmin = min(zmin, float(z[inside].min())); zmax = max(zmax, float(z[inside].max()))
        return zmin, zmax

    def save(self, path: str) -> None:
        torch.save({k: getattr(self, k) for k in ("S", "C", "U", "V", "pitch_u", "pitch_v", "n_u", "n_v",
                                                   "n_rot", "dz_rot", "z_mid", "meta")}, path)

    @classmethod
    def load(cls, path: str) -> "StepAndShootGeometry":
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def decimate(geom: StepAndShootGeometry, every: int) -> StepAndShootGeometry:
    """Keep every every-th rotation (step becomes every * dz_rot).  Apply the same selection to the sinogram rows."""
    n = (geom.n_rot - 1) // every + 1
    return StepAndShootGeometry(S=geom.S, C=geom.C, U=geom.U, V=geom.V, pitch_u=geom.pitch_u, pitch_v=geom.pitch_v, n_u=geom.n_u, n_v=geom.n_v,
                                n_rot=n, dz_rot=geom.dz_rot * every, z_mid=geom.z_mid, meta=dict(geom.meta, decimated_every=every))


@dataclass
class ProjectorSpec:
    """Everything that defines one projector of a scan, independent of the machine it runs on:
    the geometry (by asset id), the grid, the detector binning, the window height and which
    rotations are used.  Scan-specific choices live here, not in the reconstruction interface."""
    geometry_id: str
    nx: int
    B: int
    dz_slice: float = 2.0
    fov: float = 512.0
    n_win: int | None = None                 # None: what the cone covers
    rotations: list[int] | None = None       # model rotation indices used (None: all); must be equally spaced
    cache: str = "column"

    def selected(self, geom: StepAndShootGeometry) -> tuple[StepAndShootGeometry, list[int]]:
        """(geometry restricted to the selected rotations, their indices into the sinogram rows)"""
        rots = list(range(geom.n_rot)) if self.rotations is None else list(self.rotations)
        if len(rots) > 1:
            steps = {rots[i + 1] - rots[i] for i in range(len(rots) - 1)}
            assert len(steps) == 1, f"rotations must be equally spaced, got steps {steps}"
            g = StepAndShootGeometry(S=geom.S, C=geom.C, U=geom.U, V=geom.V, pitch_u=geom.pitch_u, pitch_v=geom.pitch_v, n_u=geom.n_u, n_v=geom.n_v,
                                     n_rot=len(rots), dz_rot=geom.dz_rot * steps.pop(), z_mid=geom.z_mid, meta=dict(geom.meta, rotations=rots))
        else:
            g = StepAndShootGeometry(S=geom.S, C=geom.C, U=geom.U, V=geom.V, pitch_u=geom.pitch_u, pitch_v=geom.pitch_v, n_u=geom.n_u, n_v=geom.n_v,
                                     n_rot=1, dz_rot=0.0, z_mid=geom.z_mid, meta=dict(geom.meta, rotations=rots))
        return g, rots

    def build(self, geom: StepAndShootGeometry, devices: list[str] | None = None) -> "RollingWindowOperator":
        g, rots = self.selected(geom)
        op = RollingWindowOperator(g, self.nx, self.B, self.fov, self.dz_slice, devices, self.cache, self.n_win)
        op.rotations, op.spec = rots, self
        return op

    def to_dict(self) -> dict:
        return dict(geometry_id=self.geometry_id, nx=self.nx, B=self.B, dz_slice=self.dz_slice, fov=self.fov, n_win=self.n_win,
                    rotations=self.rotations, cache=self.cache)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectorSpec":
        return cls(**{k: d[k] for k in ("geometry_id", "nx", "B", "dz_slice", "fov", "n_win", "rotations", "cache") if k in d})


# ------------------------------------------------------------------ operator
class RollingWindowOperator:
    """Projector for the whole multi-rotation volume at grid ``nx`` x ``nx`` x ``n_tot``,
    detector bin ``B``, built on ``devices`` (default: all visible GPUs)."""

    def __init__(self, geom: StepAndShootGeometry, nx: int, B: int, fov: float = 512.0,
                 dz_slice: float = 2.0, devices: list[str] | None = None, cache: str = "column", n_win: int | None = None):
        """n_win: force the window height (slices); default = what the cone covers.  Rotation j is shifted by
        geom.dz_rot which must be a whole number of slices (dz_rot / dz_slice); use :func: first
        if the scan's step is finer than the slice."""
        self.geom, self.nx, self.B, self.fov, self.dz = geom, nx, B, fov, dz_slice
        step = geom.dz_rot / dz_slice if geom.n_rot > 1 else 1.0
        assert abs(step - round(step)) < 1e-6 and round(step) >= 1, f"rotation step {geom.dz_rot} mm must be a multiple of the slice {dz_slice} mm"
        self.step = int(round(step))
        self.devices = devices or available_devices()
        self.dev = torch.device(self.devices[0])
        vox = fov / nx
        zmin, zmax = geom.ray_z_span(B, fov / 2 * math.sqrt(2))   # rays through the square volume's corners
        z0 = dz_slice * (math.floor(zmin / dz_slice) - 1)
        self.n_win = n_win or (int(math.ceil((zmax - z0) / dz_slice)) + 2)
        self.n_tot = self.n_win + (geom.n_rot - 1) * self.step
        self.R = geom.n_rot
        self.vox, self.z0 = vox, z0
        views = geom.views(B, self.dev)
        M = torch.diag(torch.tensor([vox, vox, dz_slice])).to(self.dev)
        b = torch.tensor([-(nx // 2) * vox, -(nx // 2) * vox, z0]).to(self.dev)
        t = time.time()
        if len(self.devices) > 1:
            self.A = split_voxel_projector(nx, nx, self.n_win, M, b, views, devices=self.devices,
                                           output_device=self.dev, backend="cuda", cache=cache)
        else:
            self.A = VoxelProjector3D(nx, nx, self.n_win, M, b, views, device=self.dev, backend="cuda", cache=cache)
        self._sync(); self.t_build = time.time() - t
        self.n_ray = int(self.A.n_ray)
        self.wshape, self.shape = (nx, nx, self.n_win), (nx, nx, self.n_tot)

    def _sync(self):
        for d in self.devices:
            if d.startswith("cuda"):
                torch.cuda.synchronize(d)

    # rolling application over rotations: rotation j sees slices [j, j+n_win)
    def fwd(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.A.forward_project(x[..., j * self.step:j * self.step + self.n_win].contiguous()) for j in range(self.R)])

    def adj(self, y: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(self.shape, device=self.dev)
        for j in range(self.R):
            out[..., j * self.step:j * self.step + self.n_win] += self.A.back_project(y[j]).reshape(self.wshape)
        return out

    def normal(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """A^T diag(w) A x, streamed per rotation (no [R, n_ray] intermediate)."""
        out = torch.zeros(self.shape, device=self.dev)
        for j in range(self.R):
            t = self.A.forward_project(x[..., j * self.step:j * self.step + self.n_win].contiguous()); t.mul_(w[j])
            out[..., j * self.step:j * self.step + self.n_win] += self.A.back_project(t).reshape(self.wshape)
        return out

    def window_gram(self, w_mean: torch.Tensor | None = None):
        """Gram operator of ONE window (flat in, flat out) for the eigen-preconditioner."""
        def gram(xf):
            y = self.A.forward_project(xf.reshape(self.wshape))
            if w_mean is not None:
                y = y * w_mean
            return self.A.back_project(y).reshape(-1)
        return gram

    def describe(self) -> dict:
        return dict(nx=self.nx, B=self.B, n_win=self.n_win, n_tot=self.n_tot, n_rot=self.R, n_view=self.geom.n_view,
                    n_ray=self.n_ray, vox_mm=self.vox, z0_mm=self.z0, devices=self.devices, t_build_s=round(self.t_build, 2))


# ------------------------------------------------------------------ data
def bin_sinogram(y: torch.Tensor, w: torch.Tensor, n_view: int, n_u: int, n_v: int, B: int):
    """Bin ``[R, n_view*n_u*n_v]`` sinograms (view, col, row order) by ``B`` x ``B``:
    weighted mean of valid pixels; binned weight = mean of the weights."""
    if B == 1:
        return y, w.float()
    R = y.shape[0]
    y4 = y.view(R, n_view, n_u, n_v); w4 = w.view(R, n_view, n_u, n_v).float()
    ys = F.avg_pool2d((y4 * w4).view(R * n_view, 1, n_u, n_v), B).view(R, n_view, n_u // B, n_v // B)
    ws = F.avg_pool2d(w4.view(R * n_view, 1, n_u, n_v), B).view(R, n_view, n_u // B, n_v // B)
    yb = torch.where(ws > 0, ys / ws.clamp_min(1e-12), torch.zeros_like(ys))
    return yb.reshape(R, -1).contiguous(), ws.reshape(R, -1).contiguous()


# ------------------------------------------------------------------ preconditioner, beta
def window_eigen(op: RollingWindowOperator, k: int, w_mean: torch.Tensor | None = None, **kw) -> SparseEigenDecomposition:
    dec = SparseEigenDecomposition(gram=op.window_gram(w_mean), k=k, volume_shape=op.wshape, device=op.dev)
    dec.compute_weights(method=kw.pop("method", "eigsh"), compute_projection_basis=False, verbose=False, use_tqdm=False, **kw)
    return dec


def lambda_max(op: RollingWindowOperator, w: torch.Tensor, iters: int = 10, seed: int = 0) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(op.shape, generator=g).to(op.dev)
    lam = 0.0
    for _ in range(iters):
        v = v / v.norm(); hv = op.normal(v, w); lam = float((v * hv).sum()); v = hv
    return lam


def quadratic_penalty_grad(x: torch.Tensor, beta: float) -> torch.Tensor:
    """beta * D^T D x, D = forward differences along the three axes (Neumann)."""
    g = torch.zeros_like(x)
    for dim in (0, 1, 2):
        d = torch.diff(x, dim=dim)
        gm = torch.movedim(g, dim, 0); dm = torch.movedim(d, dim, 0)
        gm[1:] += dm; gm[:-1] -= dm
    return beta * g


def make_window_preconditioner(op: RollingWindowOperator, dec: SparseEigenDecomposition):
    """Apply the window image-preconditioner P^2 slab by slab over the full volume."""
    P = dec.to_image_preconditioner()
    nw, nt = op.n_win, op.n_tot

    def Mpre(r):
        out = torch.empty_like(r)
        for s0 in range(0, nt, nw):
            s1 = min(s0 + nw, nt)
            slab = torch.zeros(op.wshape, device=r.device); slab[..., :s1 - s0] = r[..., s0:s1]
            out[..., s0:s1] = P(P(slab.reshape(-1))).reshape(op.wshape)[..., :s1 - s0]
        return out
    return Mpre


# ------------------------------------------------------------------ solver
def pcg(op: RollingWindowOperator, y: torch.Tensor, w: torch.Tensor, beta: float, iters: int,
        Mpre=None, x0: torch.Tensor | None = None, log=None, log_every: int = 16) -> tuple[torch.Tensor, dict]:
    """Preconditioned CG on (A^T W A + beta D^T D) x = A^T W y.  Returns x and a convergence log."""
    H = lambda x: op.normal(x, w) + quadratic_penalty_grad(x, beta)
    Mpre = Mpre or (lambda r: r)
    op._sync(); t0 = time.time()
    bvec = op.adj(y * w); bn = float(bvec.norm())
    x = torch.zeros(op.shape, device=op.dev) if x0 is None else x0.to(op.dev).clone()
    r = bvec - H(x) if x0 is not None else bvec.clone()
    hist = [(0, float(r.norm()) / bn, 0.0)]
    z = Mpre(r); p = z.clone(); rz = (r * z).sum()
    for it in range(iters):
        Hp = H(p); al = rz / (p * Hp).sum(); x = x + al * p; r = r - al * Hp
        z = Mpre(r); rzn = (r * z).sum(); p = z + (rzn / rz) * p; rz = rzn
        if (it + 1) % log_every == 0 or it + 1 == iters:
            op._sync(); hist.append((it + 1, float(r.norm()) / bn, time.time() - t0))
            if log:
                log(f"  iter {it + 1:4d}: rel grad {hist[-1][1]:.3e}  t {hist[-1][2]:.1f} s")
    op._sync()
    return x, dict(iters=iters, rel_grad=hist[-1][1], t_pcg_s=round(time.time() - t0, 2), history=hist)


def upsample_inplane(x: torch.Tensor, nx_from: int, nx_to: int) -> torch.Tensor:
    """Trilinear in-plane upsampling on exact voxel-centre grids (z unchanged)."""
    nzt = x.shape[2]; s = nx_to / nx_from
    j = torch.arange(nx_to, dtype=torch.float32, device=x.device)
    ci = (j / s) / (nx_from - 1) * 2 - 1
    kz = torch.arange(nzt, dtype=torch.float32, device=x.device) / (nzt - 1) * 2 - 1
    gi, gj, gk = torch.meshgrid(ci, ci, kz, indexing="ij")
    vol = x.permute(2, 1, 0)[None, None]
    grid = torch.stack([gi, gj, gk], -1).permute(2, 1, 0, 3)[None]
    return F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0].permute(2, 1, 0).contiguous()


def cascade(y1: torch.Tensor, w1: torch.Tensor, levels: list[dict], eigen_provider, log=print) -> tuple[list[tuple[int, torch.Tensor]], list[dict]]:
    """Multi-resolution cascade.  levels = [{op: RollingWindowOperator, iters, k, beta_scale}, ...] coarse to
    fine (each op built from a projector asset; all must select the same rotations); y1/w1 are the bin-1
    sinogram rows of those rotations.  eigen_provider(op, k, w_mean) -> SparseEigenDecomposition.
    Returns the volume of every level and per-level metrics."""
    vols, metrics, x = [], [], None
    for lv in levels:
        op, iters, k = lv["op"], lv["iters"], lv["k"]
        nx, B, geom = op.nx, op.B, op.geom
        t_level = time.time()
        y, w = bin_sinogram(y1.to(op.dev), w1.to(op.dev), geom.n_view, geom.n_u, geom.n_v, B)
        log(f"[{nx}] {op.describe()}")
        t = time.time(); dec = eigen_provider(op, k, w.mean(0)); t_eig = time.time() - t
        lam = dec.eigenvalues
        t = time.time(); lam_full = lambda_max(op, w); t_lam = time.time() - t
        beta = lv.get("beta_scale", 1.0) * lam_full / 1200.0
        x0 = None
        if x is not None:
            assert x.shape[2] == op.n_tot, f"cascade levels must share the slice grid ({x.shape[2]} vs {op.n_tot})"
            x0 = upsample_inplane(x.to(op.dev), prev_nx, nx)
        xs, conv = pcg(op, y, w, beta, iters, make_window_preconditioner(op, dec), x0, log=log)
        m = dict(nx=nx, B=B, k=k, iters=iters, beta=beta, lam_max=lam_full, eig_cond=float(lam.max() / lam.min()),
                 t_build_s=op.t_build, t_eig_s=round(t_eig, 2), t_lam_s=round(t_lam, 2), t_pcg_s=conv["t_pcg_s"],
                 rel_grad=conv["rel_grad"], t_level_s=round(time.time() - t_level, 2), devices=op.devices,
                 peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
                 history=conv["history"])
        log(f"[{nx}] eig {t_eig:.1f} s (cond {m['eig_cond']:.2f}), lam_max {t_lam:.1f} s, PCG {iters} it {conv['t_pcg_s']} s, level {m['t_level_s']} s")
        vols.append((nx, xs.cpu())); metrics.append(m); x, prev_nx = xs, nx
        del dec; torch.cuda.empty_cache()
    return vols, metrics
