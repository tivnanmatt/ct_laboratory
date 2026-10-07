"""Physics correction: measured transmission -> linearized water-equivalent line integrals + weights.

Per firing (one source, its active panels; columns ordered along the detector arc) the measured
air-normalized transmission is modelled as

    t = (1 - g) T + g K_G(T) + p                 T = polyenergetic primary transmission of water
        g : off-focal fraction, K_G : Gaussian halo (source-plane sd, projected by the magnification)
        p : pedestal (fraction of the air signal present under the object)

The correction inverts it: T is found by a few fixed-point sweeps (the halo is smooth, so this
converges fast), then the water beam-hardening lookup gives  L = T^{-1}(T)  in mm of water
(times mu_water at the reference energy -> 1/mm units if requested).  Statistical weights are
w = I0 * t for the post-log Gaussian approximation (variance of -log(y) ~ 1 / counts); with
unknown I0 they are relative, w = t.

This is the linear-reconstruction half of the joint raw-data fit (research-ring
recon_dev/20261005 rawjoint): the calibration parameters are *inputs* here (fitted elsewhere
or taken from the priors), so the server can correct and reconstruct without the raw scans.
Built only on ct_laboratory.physics.xray (TungstenSpectrum, OffFocalRadiation kernels, get_mu).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..attenuation.mu_utils import get_mu
from ..scatter.kernels import gaussian_matrix
from ..source.tungsten_spectrum import TungstenSpectrum, spekpy_table

__all__ = ["WaterBeamHardeningLUT", "FrameLayout", "frame_layouts_from_views", "linearize_sinogram"]


class WaterBeamHardeningLUT:
    """T(L) = sum_E w(E) exp(-mu_water(E) L) for the detected spectrum, and its inverse."""

    def __init__(self, kvp=120.0, al_mm=3.0, w_mm=0.02, e_keV=np.arange(11.0, 150.0, 2.0), detector="CsI", det_mm=0.6,
                 l_max_mm=600.0, n=2048, device="cpu"):
        kvps = np.arange(max(40.0, kvp - 20), kvp + 21, 10.0)
        edges = np.r_[e_keV - 1.0, e_keV[-1] + 1.0]
        phi = spekpy_table(kvps, edges)
        ev = e_keV * 1e3
        mu_al = get_mu("Al", ev, density=2.70)                  # get_mu already returns 1/mm
        mu_w = get_mu("W", ev, density=19.3)
        spec = TungstenSpectrum(e_keV, kvps, phi, mu_al, mu_w, kvp=kvp)
        spec.params.set_value("al_mm", al_mm); spec.params.set_value("w_mm", w_mm)
        with torch.no_grad():
            s = spec().double()
        mu_det = torch.as_tensor(get_mu("CsI", ev, density=4.51) if detector == "CsI" else get_mu(detector, ev)).double()
        det = 1.0 - torch.exp(-mu_det * det_mm)                  # energy-integrating, absorbed fraction x energy
        w = s * det * torch.as_tensor(e_keV).double(); w = w / w.sum()
        self.mu_water = torch.as_tensor(get_mu("H2O", ev, density=1.0)).double()
        self.L = torch.linspace(0.0, l_max_mm, n).double()
        self.T = (w[None, :] * torch.exp(-self.L[:, None] * self.mu_water[None, :])).sum(1)   # monotone decreasing
        self.mu_ref = float((w * self.mu_water).sum())            # effective water mu at zero thickness (1/mm)
        self.device = torch.device(device)
        self._Tf, self._Lf = self.T.flip(0).float().to(device), self.L.flip(0).float().to(device)

    def inverse(self, t: torch.Tensor) -> torch.Tensor:
        """water thickness (mm) for transmission t (clamped to the table)"""
        t = t.clamp(float(self._Tf[0]), float(self._Tf[-1]))
        i = torch.searchsorted(self._Tf, t.reshape(-1).contiguous()).clamp(1, len(self._Tf) - 1)
        t0, t1 = self._Tf[i - 1], self._Tf[i]; l0, l1 = self._Lf[i - 1], self._Lf[i]
        f = (t.reshape(-1) - t0) / (t1 - t0).clamp_min(1e-12)
        return (l0 + f * (l1 - l0)).reshape(t.shape)


class FrameLayout:
    """One firing: the views (panels) of one source, their flat ray slice and column arc positions."""

    def __init__(self, view_ids: torch.Tensor, ray_slices: list[tuple[int, int]], arc_mm: torch.Tensor, row_mm: torch.Tensor,
                 magnification: float, n_u: int, n_v: int):
        self.view_ids, self.ray_slices, self.arc_mm, self.row_mm = view_ids, ray_slices, arc_mm, row_mm
        self.magnification, self.n_u, self.n_v = magnification, n_u, n_v

    def gather(self, y_flat: torch.Tensor) -> torch.Tensor:
        """flat [..., n_ray] -> [..., n_rows, n_cols] image of this firing (columns along the arc)"""
        cols = [y_flat[..., a:b].reshape(*y_flat.shape[:-1], self.n_u, self.n_v) for a, b in self.ray_slices]
        return torch.cat(cols, dim=-2).transpose(-1, -2)         # [..., n_v rows, n_panels*n_u cols]

    def scatter(self, img: torch.Tensor, out_flat: torch.Tensor) -> None:
        x = img.transpose(-1, -2)                                  # [..., n_cols, n_rows]
        for k, (a, b) in enumerate(self.ray_slices):
            out_flat[..., a:b] = x[..., k * self.n_u:(k + 1) * self.n_u, :].reshape(*img.shape[:-2], -1)


def frame_layouts_from_views(S, C, U, V, pitch_u, pitch_v, n_u, n_v, device="cpu") -> list[FrameLayout]:
    """Group the views of one rotation by source; column arc coordinate from the panel geometry."""
    layouts = []
    src_key = torch.round(S * 1e3).to(torch.int64)
    _, inv = torch.unique(src_key, dim=0, return_inverse=True)
    r_det = float(C[:, :2].norm(dim=1).mean())
    off = torch.arange(S.shape[0] + 1) * (n_u * n_v)
    u = (torch.arange(n_u) - (n_u - 1) / 2) * pitch_u
    v = (torch.arange(n_v) - (n_v - 1) / 2) * pitch_v
    for s in range(int(inv.max()) + 1):
        ids = (inv == s).nonzero()[:, 0]
        pos = C[ids][:, None, :2] + u[None, :, None] * U[ids][:, None, :2]
        ang = torch.atan2(pos[..., 1], pos[..., 0]).reshape(-1)
        ang = torch.remainder(ang - ang[0] + math.pi, 2 * math.pi) - math.pi
        mag = float(((C[ids] - S[ids]).norm(dim=1) / S[ids].norm(dim=1)).mean())
        layouts.append(FrameLayout(ids, [(int(off[i]), int(off[i + 1])) for i in ids], (ang * r_det).to(device), v.to(device), mag, n_u, n_v))
    return layouts


def linearize_sinogram(y: torch.Tensor, w_mask: torch.Tensor, layouts: list[FrameLayout], lut: WaterBeamHardeningLUT,
                       off_focal_fraction=0.129, halo_sd_mm=61.0, focal_fwhm_mm=1.0, pedestal=0.0, n_sweeps=6,
                       I0: torch.Tensor | None = None, units="mm", t_min=1e-4, log=None):
    """y [R, n_ray] post-log line integrals (uncorrected), w_mask [R, n_ray] valid -> (L [R, n_ray], w [R, n_ray]).

    Processes rotation by rotation and firing by firing; the halo blur is one matrix product per firing.
    ``units``: 'mm' (water thickness) or '1/mm' (times lut.mu_ref)."""
    dev = y.device
    R = y.shape[0]
    t_all = torch.exp(-y).clamp(t_min, 1.5); t_all[~w_mask] = 1.0      # masked rays: air
    L = torch.empty_like(y); g, p = off_focal_fraction, pedestal
    sd_f = focal_fwhm_mm / 2.3548
    Ks = [gaussian_matrix(fl.arc_mm, torch.tensor(math.sqrt(sd_f ** 2 + halo_sd_mm ** 2) * fl.magnification)) for fl in layouts]
    for fl, K in zip(layouts, Ks):
        t = fl.gather(t_all)                                            # [R, rows, cols]
        T = t.clone()
        for _ in range(n_sweeps):
            G = 1.0 + (T.mean(-2, keepdim=True) - 1.0) @ K.T            # row-averaged halo, air outside
            T = ((t - p - g * G) / (1.0 - g)).clamp(t_min, 1.5)
        fl.scatter(lut.inverse(T), L)
    if units == "1/mm":
        L = L * lut.mu_ref
    w = (t_all if I0 is None else I0 * t_all) * w_mask.float()
    L = L * w_mask.float()
    if log:
        log(f"linearize: {R} rotations, {len(layouts)} firings, g={g}, halo sd {halo_sd_mm} mm, pedestal {p}, mu_ref {lut.mu_ref:.5f}/mm")
    return L, w
