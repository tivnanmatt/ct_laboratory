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

import contextlib
import math
import gc, time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..sparse_eigen_preconditioner import SparseEigenDecomposition
from ..tomography import VoxelProjector3D, split_voxel_projector
from ..tomography.voxel_projector_3d_module import make_views
from ..workflow.gpus import available_devices

__all__ = ["StepAndShootGeometry", "ProjectorSpec", "decimate", "RollingWindowOperator", "bin_sinogram", "window_eigen", "make_window_preconditioner", "resample_volume", "resample_physical", "volume_box",
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
    z_center: float | None = None            # centre of the (forced) window in mm; None: centre of the cone's z extent

    def selected(self, geom: StepAndShootGeometry) -> tuple[StepAndShootGeometry, list[int]]:
        """(geometry restricted to the selected rotations, their indices into the sinogram rows)"""
        rots = list(range(geom.n_rot)) if self.rotations is None else list(self.rotations)
        if len(rots) > 1:
            steps = {rots[i + 1] - rots[i] for i in range(len(rots) - 1)}
            assert len(steps) == 1, f"rotations must be equally spaced, got steps {steps}"
            g = StepAndShootGeometry(S=geom.S, C=geom.C, U=geom.U, V=geom.V, pitch_u=geom.pitch_u, pitch_v=geom.pitch_v, n_u=geom.n_u, n_v=geom.n_v,
                                     n_rot=len(rots), dz_rot=geom.dz_rot * steps.pop(), z_mid=geom.z_mid, meta=dict(geom.meta, rotations=rots, dz_rot_model=geom.dz_rot))
        else:
            g = StepAndShootGeometry(S=geom.S, C=geom.C, U=geom.U, V=geom.V, pitch_u=geom.pitch_u, pitch_v=geom.pitch_v, n_u=geom.n_u, n_v=geom.n_v,
                                     n_rot=1, dz_rot=0.0, z_mid=geom.z_mid, meta=dict(geom.meta, rotations=rots, dz_rot_model=geom.dz_rot))
        return g, rots

    def build(self, geom: StepAndShootGeometry, devices: list[str] | None = None) -> "RollingWindowOperator":
        g, rots = self.selected(geom)
        op = RollingWindowOperator(g, self.nx, self.B, self.fov, self.dz_slice, devices, self.cache, self.n_win, self.z_center)
        op.rotations, op.spec = rots, self
        return op

    def to_dict(self) -> dict:
        return dict(geometry_id=self.geometry_id, nx=self.nx, B=self.B, dz_slice=self.dz_slice, fov=self.fov, n_win=self.n_win,
                    rotations=self.rotations, cache=self.cache, z_center=self.z_center)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectorSpec":
        return cls(**{k: d[k] for k in ("geometry_id", "nx", "B", "dz_slice", "fov", "n_win", "rotations", "cache", "z_center") if k in d})


# ------------------------------------------------------------------ operator
class RollingWindowOperator:
    """Projector for the whole multi-rotation volume at grid ``nx`` x ``nx`` x ``n_tot``,
    detector bin ``B``, built on ``devices`` (default: all visible GPUs)."""

    def __init__(self, geom: StepAndShootGeometry, nx: int, B: int, fov: float = 512.0,
                 dz_slice: float = 2.0, devices: list[str] | None = None, cache: str = "column", n_win: int | None = None,
                 z_center: float | None = None):
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
        if n_win:                                   # forced window: centred on z_center (default: the cone's z extent)
            zc = (zmin + zmax) / 2 if z_center is None else z_center
            z0 = dz_slice * round(zc / dz_slice) - dz_slice * (n_win // 2)
            self.n_win = n_win
        else:
            z0 = dz_slice * (math.floor(zmin / dz_slice) - 1)
            self.n_win = int(math.ceil((zmax - z0) / dz_slice)) + 2
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
        else:   # the SF kernels launch on the CURRENT device: build (and use) under this projector's device
            with torch.cuda.device(self.dev) if self.dev.type == "cuda" else contextlib.nullcontext():
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

    def window_gram(self, w_mean: torch.Tensor | None = None, beta: float = 0.0, scale: torch.Tensor | None = None):
        """Window operator for the eigen-preconditioner (flat in, flat out):
            S^-1 (A^T diag(w_mean) A + beta D^T D) S^-1      with S = sqrt(scale) (None: no scaling)
        i.e. the (optionally sensitivity-normalized) REGULARIZED Hessian of one window."""
        Ss = None if scale is None else scale.reshape(-1).sqrt()
        def gram(xf):
            x = xf if Ss is None else xf / Ss
            y = self.A.forward_project(x.reshape(self.wshape))
            if w_mean is not None:
                y = y * w_mean
            g = self.A.back_project(y).reshape(-1)
            if beta:
                g = g + quadratic_penalty_grad(x.reshape(self.wshape), beta).reshape(-1)
            return g if Ss is None else g / Ss
        return gram

    def sensitivity(self, w_mean: torch.Tensor | None = None, beta: float = 0.0) -> torch.Tensor:
        """Diagonal of the regularized window Hessian: A^T w_mean (+ 6 beta, the Laplacian diagonal). One back projection."""
        ones = torch.ones(self.n_ray, device=self.dev) if w_mean is None else w_mean
        return self.A.back_project(ones).reshape(-1) + 6.0 * beta

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
def window_eigen(op: RollingWindowOperator, k: int, w_mean: torch.Tensor | None = None, beta: float = 0.0,
                 scale: torch.Tensor | None = None, **kw) -> SparseEigenDecomposition:
    """Top-k eigenpairs of the window operator (see window_gram): scaled regularized Hessian when beta/scale are given."""
    dec = SparseEigenDecomposition(gram=op.window_gram(w_mean, beta, scale), k=k, volume_shape=op.wshape, device=op.dev)
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


def make_window_preconditioner(op: RollingWindowOperator, dec: SparseEigenDecomposition, scale: torch.Tensor | None = None):
    """Apply the window image-preconditioner slab by slab over the full volume.  With ``scale`` (the diagonal the basis
    was computed under) the preconditioner is S^-1 P^2 S^-1, S = sqrt(scale): one eigen-filtered step in the scaled space."""
    P = dec.to_image_preconditioner()
    nw, nt = op.n_win, op.n_tot
    Ss = None if scale is None else scale.reshape(op.wshape).sqrt()

    def Mpre(r):
        out = torch.empty_like(r)
        for s0 in range(0, nt, nw):
            s1 = min(s0 + nw, nt)
            slab = torch.zeros(op.wshape, device=r.device); slab[..., :s1 - s0] = r[..., s0:s1]
            if Ss is not None:
                slab = slab / Ss
            z = P(P(slab.reshape(-1))).reshape(op.wshape)
            if Ss is not None:
                z = z / Ss
            out[..., s0:s1] = z[..., :s1 - s0]
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
    return resample_volume(x, (nx_to, nx_to, x.shape[2]))


def resample_volume(x: torch.Tensor, shape_to: tuple[int, int, int]) -> torch.Tensor:
    """Trilinear resampling between two grids that cover the SAME physical box (voxel centres at
    (i + 0.5) / n of the box): used to warm-start a finer cascade level, in-plane and/or axially."""
    if tuple(x.shape) == tuple(shape_to):
        return x
    vol = x.permute(2, 1, 0)[None, None]                                   # [1,1,Z,Y,X]
    out = F.interpolate(vol, size=(shape_to[2], shape_to[1], shape_to[0]), mode="trilinear", align_corners=False)
    return out[0, 0].permute(2, 1, 0).contiguous()


def volume_box(op) -> dict:
    """Physical box of an operator's volume: voxel centres at x0 + (i + 0.5) * vox in-plane and z0 + (k + 0.5) * dz
    axially (absolute z: the rolling window is placed in the frame of the first selected model rotation)."""
    rots = list(getattr(op, "rotations", [0]) or [0]); g = op.geom
    d0 = g.meta.get("dz_rot_model") or (g.dz_rot / (rots[1] - rots[0]) if len(rots) > 1 else 0.0)
    return dict(x0=-(op.nx // 2) * op.vox, vox=op.vox, nx=op.nx, z0=op.z0 + rots[0] * d0, dz=op.dz, nz=op.n_tot)


def _interp_matrix(c_to: torch.Tensor, start: float, step: float, n: int) -> torch.Tensor:
    """[len(c_to), n] linear-interpolation weights from a grid with centres start + (i + 0.5) * step; zero outside its box,
    clamped to the outermost centres within the box."""
    u = (c_to - start) / step - 0.5; inside = (u >= -0.5) & (u <= n - 0.5)
    u = u.clamp(0, n - 1); i0 = u.floor().long().clamp(max=max(n - 2, 0)); f = (u - i0).clamp(0, 1)
    W = torch.zeros(len(c_to), n, dtype=torch.float32)
    r = torch.arange(len(c_to)); W[r, i0] += (1 - f); W[r, (i0 + 1).clamp(max=n - 1)] += f
    W[~inside] = 0.0; return W


def resample_physical(x: torch.Tensor, box_from: dict, box_to: dict) -> torch.Tensor:
    """Warm-start resampling between cascade levels that may cover DIFFERENT physical boxes (field of view, slice size,
    axial window): separable linear interpolation by physical position, zero outside the source box."""
    cx = box_to["x0"] + (torch.arange(box_to["nx"]) + 0.5) * box_to["vox"]; cz = box_to["z0"] + (torch.arange(box_to["nz"]) + 0.5) * box_to["dz"]
    Wx = _interp_matrix(cx, box_from["x0"], box_from["vox"], x.shape[0]); Wz = _interp_matrix(cz, box_from["z0"], box_from["dz"], x.shape[2])
    out = torch.einsum("ai,ijk->ajk", Wx, x.float().cpu()); out = torch.einsum("bj,ajk->abk", Wx, out); out = torch.einsum("ck,abk->abc", Wz, out)
    return out.contiguous()


def cascade(y1: torch.Tensor, w1: torch.Tensor, levels: list[dict], eigen_provider, log=print) -> tuple[list[tuple[int, torch.Tensor]], list[dict]]:
    """Multi-resolution cascade.  levels = [{op: RollingWindowOperator, iters, k, beta_scale}, ...] coarse to
    fine (each op built from a projector asset; levels may select different stations); y1/w1 are the bin-1
    sinogram rows of ALL stations (or of exactly this level's stations).  eigen_provider(op, k, w_mean) -> SparseEigenDecomposition.
    Returns the volume of every level and per-level metrics."""
    vols, metrics, x, box_prev = [], [], None, None
    for lv in levels:
        op = lv["op"] if lv.get("op") is not None else lv["build"]()                   # ops may be built lazily (one level resident at a time)
        iters, k = lv["iters"], lv["k"]
        nx, B, geom = op.nx, op.B, op.geom
        t_level = time.time()
        rows = op.rotations if y1.shape[0] != len(op.rotations) else slice(None)        # y1/w1: all stations (bin 1) -> this level's stations
        y, w = bin_sinogram(y1[rows], w1[rows], geom.n_view, geom.n_u, geom.n_v, B)    # bin on the CPU (bin-1 temporaries are GBs), then move
        y, w = y.to(op.dev), w.to(op.dev)
        log(f"[{nx}] {op.describe()}")
        t = time.time(); lam_full = lambda_max(op, w); t_lam = time.time() - t
        beta = lv.get("beta_scale", 1.0) * lam_full / 1200.0
        scale = op.sensitivity(w.mean(0), beta) if lv.get("scaling", "sensitivity") == "sensitivity" else None
        t = time.time(); dec = eigen_provider(op, k, w.mean(0), beta, scale); t_eig = time.time() - t
        lam = dec.eigenvalues
        box = volume_box(op)                                                    # levels may cover different boxes (fov, slices, window):
        x0 = None if x is None else resample_physical(x, box_prev, box).to(op.dev)   # warm start by physical position (fixed 2026-10-10)
        xs, conv = pcg(op, y, w, beta, iters, make_window_preconditioner(op, dec, scale), x0, log=log)
        m = dict(nx=nx, B=B, k=k, iters=iters, beta=beta, lam_max=lam_full, eig_cond=float(lam.max() / lam.min()),
                 t_build_s=op.t_build, t_eig_s=round(t_eig, 2), t_lam_s=round(t_lam, 2), t_pcg_s=conv["t_pcg_s"],
                 rel_grad=conv["rel_grad"], t_level_s=round(time.time() - t_level, 2), devices=op.devices,
                 peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
                 history=conv["history"], rotations=list(op.rotations))
        log(f"[{nx}] eig {t_eig:.1f} s (cond {m['eig_cond']:.2f}), lam_max {t_lam:.1f} s, PCG {iters} it {conv['t_pcg_s']} s, level {m['t_level_s']} s")
        vols.append((nx, xs.cpu())); metrics.append(m); x = xs.cpu(); box_prev = box
        del dec, xs, y, w, scale, x0; lv["op"] = None; del op; gc.collect(); torch.cuda.empty_cache()   # free this level before the next
    return vols, metrics
