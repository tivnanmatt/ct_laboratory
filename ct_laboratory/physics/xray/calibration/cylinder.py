"""Per-source X-ray system calibration on an analytic cylinder phantom (the "cylinder-basis" fit).

Object model: one or more infinite z-aligned cylinders of KNOWN material, optionally clipped to a z slab, with fitted radius, density
scale and axis position (Gaussian priors).  Chords are exact (tomography.analytic_cylinder), averaged over NSUB sub-rays across the
pixel along the arc so the polyenergetic partial-volume at the rim is right; the sub-ray spread feeds an errors-in-variables term.
Optional fixed extra water-equivalent path per pixel (e.g. a patient table projected from a reconstruction).

Measurement model (AirNormalizedProjectionModel, binned scatter):
    t = out_k (1 + eps_j) [ (1 - a_g,j) ((1 - g) P + g K_G P) + a_ph,j ],      P = sum_E w_s(E) <exp(-mu(E) L)>_sub
    w_s   per-source detected spectrum: TungstenSpectrum (kVp, Al, spline free; W / heel fixed) x ScintillatorDetector
    a_g   gain-scan scatter per 8 x 8 px bin, very strong Laplacian (room scatter, very low frequency)
    a_ph  phantom-scan scatter = a_g + delta per bin, weaker Laplacian (object-dependent change)
    eps_j per-pixel gain error shared by all stations, prior sd sigma_e (closed form, PixelGainError)
    out_k per-station output factor (profiled, differentiable)
Likelihood: AirNormalizedGaussian - Poisson + read noise + model-error floor + edge jitter + (dP/dL)^2 sigma_L^2.
Outliers (|z| > 6 after a quarter of the steps) are masked.  Optimizer: Adam + cosine schedule on whitened parameters.

Outputs per source (CylinderCalibration.result): fitted parameters, per-pixel correction maps (a_g, a_ph, eps), detected spectrum,
and the projection-domain QA of the corrected data in HU-equivalent units (1000 (L_est - L_true) / L_true): CT-number accuracy,
centre and peripheral (x = +-0.7 R) ROIs, uniformity (max |peripheral - centre|), diameter profile vs signed ray offset x / R, 1-sigma coverage.

Joint mode: several datasets (object scans, e.g. the 7 calcium cylinders) calibrate one source together - spectrum, detector, off-focal,
gain-scan scatter and per-pixel eps are shared; phantom-scan scatter, geometry and output are per dataset (CylinderCalibration(cfg, [phantoms], [datas])).

This is the method of recon_dev/20261005_multirot_active_window/steps/acr_ctlab_fit.py (2026-10-07), made reusable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from ..parameters import PriorParameters
from ..source import TungstenSpectrum
from ..detector import ScintillatorDetector, AirNormalizedGaussian, PixelGainError
from ..scatter import OffFocalRadiation, FocalSpotBlur, BinnedAdditiveScatter, ScatterSpectrum, module_bin_index, bin_grid
from ..xray_system import AirNormalizedProjectionModel
from ....tomography.analytic_cylinder import cylinder_chords


@dataclass
class CylinderPhantom:
    """z-aligned cylinder(s): radius (mm), density scale, axis (cx, cy); mu [n_E] linear attenuation of the material at density 1 (1/mm)."""
    radius: float
    cx: float
    cy: float
    mu: torch.Tensor                      # [n_E] 1/mm
    density: float = 1.0
    z_slab: tuple[float, float] = (-1e4, 1e4)
    radius_sd: float = 1.0
    density_sd: float = 0.02
    axis_sd: float = 0.5
    require_inside_slab: bool = True      # drop rays whose chord leaves the slab (other phantom sections)


@dataclass
class FiringData:
    """One source firing at native pixels, image layout [n_rows, n_cols] (columns ordered along the arc), NK stations.

    t          [NK, R, C]  air-normalized measurement (y - kappa) / I0
    I0, kappa  [R, C]      air signal (photon equivalents) and shifted-Poisson offset
    good       [NK, R, C]  valid samples
    det_pos    [R, C, 3]   pixel centres; det_normal [R, C, 3]; src [3]
    sub_pos    [R, C, NSUB, 3] sub-ray end points (along the arc)
    arc_mm [C], row_mm [R]; module/col/row index [R, C] for binning; z_offsets [NK] object z shift per station (mm)
    extra_L    [NK, R, C]  fixed extra water-equivalent path (table), may be zeros; extra_L_sd_frac relative uncertainty
    fluence_ratio [R, C]  F_bar / F_j (smooth air fluence), ones if unknown
    """
    t: torch.Tensor; I0: torch.Tensor; kappa: torch.Tensor; good: torch.Tensor
    det_pos: torch.Tensor; det_normal: torch.Tensor; src: torch.Tensor; sub_pos: torch.Tensor
    arc_mm: torch.Tensor; row_mm: torch.Tensor; module: torch.Tensor; col: torch.Tensor; row: torch.Tensor
    z_offsets: torch.Tensor
    extra_L: torch.Tensor | None = None
    extra_L_sd_frac: float = 0.2
    fluence_ratio: torch.Tensor | None = None
    magnification: float = 0.74
    cone_offset_deg: float = 0.0


@dataclass
class CylinderCalibrationConfig:
    energies_keV: np.ndarray
    spek_kvps: np.ndarray
    spek_phi: np.ndarray                  # [n_kv, n_E] unfiltered spectra
    mu_al: np.ndarray; mu_w: np.ndarray; mu_scint: np.ndarray
    kvp: float = 120.0
    al_mm: float = 17.0; w_mm: float = 0.013; heel: float = 0.0; spline: list = field(default_factory=lambda: [0.0] * 4); csi_mm: float = 0.6
    off_focal_fraction: float = 0.129; halo_sd_mm: float = 61.0
    room_fraction: float = 0.035
    floor: float = 0.002; jitter_px: float = 0.195
    eps_sigma: float = 0.01
    laplacian_gain: float = 5e7; laplacian_phantom: float = 1.25e6
    phantom_sd: float = 0.05
    bin_px: int = 8
    steps: int = 200; lr: float = 0.05; outlier_z: float = 6.0
    free_spectrum: bool = True            # kVp, Al, spline per source
    scatter_spectrum: bool = True
    qa_roi_half_width: float = 0.11       # ROI half width / R
    qa_peripheral_x: float = 0.7          # peripheral ROI centres at +-x / R (small phantoms: keep clear of the rim blur, e.g. 0.5)


class _Block:
    """one dataset (object scan) of the source being calibrated: its phantom, data, geometry parameters and masks"""

    def __init__(self, phantom: CylinderPhantom, data: FiringData, dev):
        self.ph, self.D = phantom, data
        self.mu_obj = phantom.mu.to(dev).float()
        self.geo_mu = torch.tensor([phantom.radius, phantom.density, phantom.cx, phantom.cy], device=dev)
        self.geo_sd = torch.tensor([phantom.radius_sd, phantom.density_sd, phantom.axis_sd, phantom.axis_sd], device=dev)
        self.gz = torch.zeros(4, device=dev, requires_grad=True)
        NK = data.t.shape[0]
        zoff = torch.zeros(NK, 1, 1, 1, 3, device=dev); zoff[..., 2] = data.z_offsets[:, None, None, None]
        self.src_all = (data.src + zoff).expand((NK,) + tuple(data.sub_pos.shape)).contiguous(); self.dst_all = data.sub_pos[None] + zoff
        self.extra = data.extra_L if data.extra_L is not None else torch.zeros_like(data.t)
        with torch.no_grad():
            Lin, Lout = self.chords(self.geo_mu)
            self.inside = (Lout.max(-1).values < 0.5) if phantom.require_inside_slab else torch.ones_like(data.good)
        self.mask_out = torch.ones_like(data.good)
        self.gy = torch.zeros_like(data.t); self.gy[..., 1:-1] = (data.t[..., 2:] - data.t[..., :-2]).abs() / 2

    def geo(self): return self.geo_mu + self.geo_sd * self.gz

    def chords(self, geo):
        lo, hi = self.ph.z_slab
        Lin, Lb, La = cylinder_chords(self.src_all, self.dst_all, geo[2], geo[3], geo[0], z_slabs=((lo, hi), (-1e4, lo), (hi, 1e4)))
        return Lin, Lb + La


class CylinderCalibration:
    """Calibration of ONE source from one or several cylinder datasets.

    Shared by all datasets: spectrum (kVp, Al, spline), detector, off-focal, gain-scan scatter a_g (one gain scan serves every object),
    per-pixel gain error eps (profiled jointly over every station of every dataset), likelihood floor.
    Per dataset (object dependent): phantom-scan scatter a_ph (one binned field per dataset), cylinder geometry (R, rho, axis), per-station output.
    All datasets must share the pixel layout of this source's firing (same geometry, same active modules).
    ``phantom`` / ``data`` may be single objects (one dataset, the original behaviour) or lists.
    """

    def __init__(self, cfg: CylinderCalibrationConfig, phantom, data, device="cuda:0"):
        phs = phantom if isinstance(phantom, (list, tuple)) else [phantom]
        ds = data if isinstance(data, (list, tuple)) else [data]
        assert len(phs) == len(ds) and len(ds) >= 1
        self.single = not isinstance(data, (list, tuple))
        self.cfg, self.dev = cfg, torch.device(device)
        c, dev = cfg, self.dev
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.E = f32(c.energies_keV)
        D0 = ds[0]; R_, C_ = D0.t.shape[1:]
        for D in ds[1:]: assert tuple(D.t.shape[1:]) == (R_, C_), "all datasets of one source need the same pixel layout"
        src = TungstenSpectrum(c.energies_keV, c.spek_kvps, c.spek_phi, c.mu_al, c.mu_w, kvp=c.kvp,
                               priors=dict(kvp=(c.kvp, 1.0), al_mm=(c.al_mm, math.log(2.0)), w_mm=(c.w_mm, math.log(5.0)), heel=(c.heel, 0.2)))
        for k_, v_ in (("kvp", c.kvp), ("al_mm", c.al_mm), ("w_mm", c.w_mm), ("heel", c.heel)): src.params.set_value(k_, v_)
        src.params.set_prior("spline", torch.tensor(c.spline), 0.1).set_value("spline", torch.tensor(c.spline))
        src.params.fix("w_mm", "heel")
        if not c.free_spectrum: src.params.fix()
        det = ScintillatorDetector(c.energies_keV, c.mu_scint, thickness_mm=c.csi_mm); det.params.fix()
        off = OffFocalRadiation(fraction=c.off_focal_fraction, halo_sd_mm=c.halo_sd_mm); off.params.fix()
        self.lik = AirNormalizedGaussian(floor=c.floor, jitter_px=c.jitter_px).to(dev); self.lik.params.fix("jitter_px")
        BI = module_bin_index(D0.module, D0.col, D0.row, bin_px=c.bin_px); nb = int(BI.max()) + 1
        colpos = torch.argsort(torch.argsort(D0.arc_mm)).to(dev)
        grid = bin_grid(BI, colpos[None].expand(R_, C_), bins_rows=(int(D0.row.max()) + 1) // c.bin_px, bin_row_index=D0.row // c.bin_px)
        self.binned = BinnedAdditiveScatter(BI, nb, n_phantom_fields=len(ds), gain=c.room_fraction, gain_sd=0.03, phantom_sd=c.phantom_sd, gain_flat=False,
                                            phantom_relative_to_gain=True, grid=grid, laplacian_gain=c.laplacian_gain, laplacian_phantom=c.laplacian_phantom)
        self.model = AirNormalizedProjectionModel(src, det, off, binned=self.binned, focal_blur=FocalSpotBlur(),
                                                  scatter_spectrum=ScatterSpectrum() if c.scatter_spectrum else None).to(dev)
        self.epi = PixelGainError(c.eps_sigma)
        dvec = D0.det_pos - D0.src; dn = dvec.norm(dim=-1)
        self.cos_inc = (dvec * D0.det_normal).sum(-1).abs() / dn
        self.cone = torch.rad2deg(torch.atan2(dvec[..., 2], dvec[..., :2].norm(dim=-1))) - D0.cone_offset_deg
        self.fr = D0.fluence_ratio if D0.fluence_ratio is not None else torch.ones_like(D0.I0)
        self.blocks = [_Block(ph, D, dev) for ph, D in zip(phs, ds)]
        self.snaps, self.losses = [], []

    # backwards-compatible accessors (single dataset)
    @property
    def D(self): return self.blocks[0].D
    @property
    def gz(self): return self.blocks[0].gz
    def geo(self): return self.blocks[0].geo()

    def _block_forward(self, b: _Block, field: int):
        D, geo = b.D, b.geo()
        Lin, _ = b.chords(geo); Leq = geo[1] * Lin + b.extra[..., None]
        o = self.model(Leq[..., None] * b.mu_obj, self.cos_inc, self.cone, D.arc_mm, D.row_mm, D.magnification, fluence_ratio=self.fr, phantom_field=field)
        T = o["T"]; mu_eff = (o["TE"] * o["w"] * b.mu_obj).sum(-1) / T.clamp(min=1e-9)
        Lm = Lin.mean(-1); sub_var = (geo[1] * Lin).var(-1) if Lin.shape[-1] > 1 else torch.zeros_like(T)
        bimp = torch.sqrt((geo[0] ** 2 - (Lm / 2) ** 2).clamp(min=1e-6)); dLdb = (bimp / torch.sqrt((geo[0] ** 2 - bimp ** 2).clamp(min=1.0))) * 2 * 0.3
        sigL2 = sub_var + dLdb ** 2 * (Lm > 0.1) + (D.extra_L_sd_frac * b.extra) ** 2
        msk = D.good & b.inside & b.mask_out
        v0 = self.lik.variance(D.t, D.I0, D.kappa, b.gy); z_ = torch.zeros_like(T)
        sc = (torch.where(msk, D.t * o["t"] / v0, z_).sum((1, 2)) / torch.where(msk, o["t"] ** 2 / v0, z_).sum((1, 2)).clamp(min=1e-9))[:, None, None]
        v = v0 + (sc * o["primary_weight"] * mu_eff * T) ** 2 * sigL2
        return dict(m=sc * o["t"], v=v, sc=sc, mask=msk, T=T, w=o["w"], Lin=Lm, Leq=Leq.mean(-1), G=sc * o["off_focal"], S=sc * o["scatter"] * torch.ones_like(T),
                    pw=o["primary_weight"] * torch.ones_like(T), ag=o["gain_scatter"] * torch.ones_like(D.I0), sigL=sigL2.detach().sqrt(), geo=geo)

    def forward_all(self):
        ps = [self._block_forward(b, i) for i, b in enumerate(self.blocks)]
        t = torch.cat([b.D.t for b in self.blocks]); m = torch.cat([p["m"] for p in ps]); v = torch.cat([p["v"] for p in ps]); mk = torch.cat([p["mask"] for p in ps])
        eps = self.epi.profile(t, m, v, mk)                               # shared by every station of every dataset (detector pixel property)
        nll = self.epi.neg_log_prior(eps)
        for b, p in zip(self.blocks, ps):
            p["m"] = p["m"] * (1 + eps); p["eps"] = eps
            nll = nll + self.lik.nll(b.D.t, p["m"], p["v"], p["mask"])
        return nll, ps

    def forward(self):
        nll, ps = self.forward_all()
        return nll, (ps[0] if self.single else ps)

    def loss(self):
        nll, ps = self.forward_all()
        return nll + self.model.neg_log_prior() + self.lik.params.neg_log_prior() + sum(0.5 * (b.gz.double() ** 2).sum() for b in self.blocks), ps

    def fit(self, log=print, snapshot_every=None):
        c = self.cfg
        pars = [p for p in self.model.parameters() if p.requires_grad] + [p for p in self.lik.parameters() if p.requires_grad]
        gzs = [b.gz for b in self.blocks]
        opt = torch.optim.Adam([dict(params=pars, lr=c.lr), dict(params=gzs, lr=c.lr)]); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c.steps)
        every = snapshot_every or max(1, c.steps // 40)
        for step in range(1, c.steps + 1):
            opt.zero_grad(); f, ps = self.loss(); f = torch.nan_to_num(f, nan=1e30, posinf=1e30); f.backward()
            for q in pars + gzs:
                if q.grad is not None: q.grad.nan_to_num_(0.0, 0.0, 0.0)
            opt.step(); sch.step(); self.losses.append(float(f.detach()))
            if step == c.steps // 4:
                with torch.no_grad():
                    _, ps = self.forward_all(); nbad = 0
                    for b, p in zip(self.blocks, ps):
                        o_ = (((b.D.t - p["m"]) / p["v"].sqrt()).abs() > c.outlier_z) & p["mask"]; b.mask_out &= ~o_; nbad += int(o_.sum())
                log(f"   outlier rejection: {nbad} samples masked")
            if step % every == 0 or step == c.steps:
                with torch.no_grad(): _, ps = self.forward_all()
                self.snaps.append(dict(step=step, geo=[p["geo"].detach().cpu() for p in ps], summary=self.summary(), loss=float(f)))
        return self

    def summary(self):
        out = {}
        for ps in list(self.model.parameter_sets().values()) + [self.lik.params]:
            for n in ps.names(free_only=True): out[ps._meta[n]["label"]] = ps.summary()[ps._meta[n]["label"]]
        return out

    @torch.no_grad()
    def _block_result(self, b: _Block, p: dict):
        D = b.D; geo = p["geo"]; mu = b.mu_obj
        Lg = torch.linspace(0, 320, 641, device=self.dev)
        Tc = ((D.t / (1 + p["eps"]) - p["G"] - p["S"]) / (p["sc"] * p["pw"])).clamp(1e-5, 1.5)
        curve = torch.einsum("rce,le->rcl", p["w"], torch.exp(-mu[None] * Lg[:, None]))    # [R, C, nL] (per-pixel spectrum, same for all stations)
        j = torch.stack([(curve > Tc[k][..., None]).sum(-1) for k in range(Tc.shape[0])]).clamp(1, len(Lg) - 1)       # station by station (memory)
        c0 = torch.stack([curve.gather(-1, (j[k] - 1)[..., None])[..., 0] for k in range(Tc.shape[0])]); c1 = torch.stack([curve.gather(-1, j[k][..., None])[..., 0] for k in range(Tc.shape[0])])
        Lest = Lg[j - 1] + (c0 - Tc) / (c0 - c1).clamp(min=1e-9) * (Lg[1] - Lg[0]) - b.extra
        mu_e = (p["w"] * mu * torch.exp(-mu * Lest.clamp(min=0)[..., None])).sum(-1) / Tc.clamp(min=1e-6)
        sdL = torch.sqrt(p["v"]) / (p["sc"] * p["pw"]) / (mu_e * Tc).clamp(min=1e-9)
        g_ = p["mask"] & (p["Lin"] > 1)
        # signed ray offset x from the cylinder axis (transaxial): |x| from the chord, sign from the side of the axis the ray passes
        Lt = geo[1] * p["Lin"]; absx = torch.sqrt((geo[0] ** 2 - (p["Lin"] / 2) ** 2).clamp(min=0))
        d2 = (b.dst_all.mean(-2) - b.src_all.mean(-2))[..., :2]; w2 = torch.stack([geo[2], geo[3]]) - b.src_all.mean(-2)[..., :2]
        sgn = torch.sign(d2[..., 0] * w2[..., 1] - d2[..., 1] * w2[..., 0]); x = (sgn * absx / geo[0])[g_].cpu().numpy()
        # HU-equivalent line-integral error: 1000 (L_est - L_true) / L_true = mean attenuation error along the ray
        hu = (1000 * (Lest - Lt) / Lt.clamp(min=1e-3))[g_].cpu().numpy(); hu_sd = (1000 * sdL / Lt.clamp(min=1e-3))[g_].cpu().numpy()
        w = 1 / np.maximum(hu_sd, 0.5) ** 2
        def roi(m_):
            if m_.sum() < 20: return dict(mean_hu=float("nan"), se_hu=float("nan"), n=int(m_.sum()))
            ww = w[m_]; mm = (ww * hu[m_]).sum() / ww.sum()
            return dict(mean_hu=float(mm), se_hu=float(np.sqrt((ww ** 2 * (hu[m_] - mm) ** 2).sum()) / ww.sum()), n=int(m_.sum()))
        edges = np.linspace(-1.0, 1.0, 41); prof = [roi((x >= a) & (x < c)) for a, c in zip(edges[:-1], edges[1:])]
        H, XP = self.cfg.qa_roi_half_width, self.cfg.qa_peripheral_x     # ACR layout: 11 mm ROIs, peripheral centres 70 mm from the axis of a 100 mm radius phantom
        rois = dict(centre=roi(np.abs(x) < H), peripheral_left=roi(np.abs(x + XP) < H), peripheral_right=roi(np.abs(x - XP) < H))
        uni = max(abs(rois[k]["mean_hu"] - rois["centre"]["mean_hu"]) for k in ("peripheral_left", "peripheral_right"))
        acc = roi(np.ones_like(x, dtype=bool))
        nz = (hu - acc["mean_hu"] * 0) / np.maximum(hu_sd, 1e-3)
        return dict(geo=[float(v) for v in geo],
                    qa=dict(units="HU-equivalent line-integral error 1000 (L_est - L_true) / L_true",
                            accuracy=acc, roi=rois, uniformity_hu=float(uni),
                            diameter_profile=dict(x_over_R=(0.5 * (edges[1:] + edges[:-1])).tolist(), mean_hu=[q["mean_hu"] for q in prof], se_hu=[q["se_hu"] for q in prof]),
                            coverage_1sigma=float(np.mean(np.abs(nz) < 1))),
                    n_outliers=int((~b.mask_out).sum()), aph=(p["S"][0] / p["sc"][0]).cpu(),
                    corrected=dict(L_est=Lest[g_].cpu(), L_true=Lt[g_].cpu(), sd=sdL[g_].cpu(), x_over_R=torch.from_numpy(x)))

    @torch.no_grad()
    def result(self):
        """shared parameters / maps / spectrum + per-dataset geometry, phantom scatter and corrected-data length test.
        Single dataset: the per-dataset entries are merged into the top level (original layout)."""
        _, ps = self.forward_all(); p0 = ps[0]
        mk = torch.stack([p["mask"].any(0) for p in ps]).any(0)
        w0 = p0["w"][mk].mean(0); w0 = w0 / w0.sum()
        shared = dict(params=self.summary(), losses=self.losses, snaps=self.snaps, eps_rms=float(p0["eps"][mk].pow(2).mean().sqrt()),
                      maps=dict(ag=p0["ag"].cpu(), eps=p0["eps"].cpu(), mask=mk.cpu()), spectrum=dict(energies_keV=self.E.cpu(), w_src=w0.cpu()),
                      off_focal=dict(fraction=float(self.model.off_focal.fraction), halo_sd_mm=float(self.model.off_focal.params["halo_sd_mm"]),
                                     focal_sd_mm=self.model.off_focal.focal_sd_mm, mag=self.blocks[0].D.magnification))
        per = [self._block_result(b, p) for b, p in zip(self.blocks, ps)]
        if self.single:
            r = dict(shared, **per[0]); r["maps"] = dict(shared["maps"], aph=per[0].pop("aph")); r.pop("aph", None); return r
        return dict(shared, datasets=per)
