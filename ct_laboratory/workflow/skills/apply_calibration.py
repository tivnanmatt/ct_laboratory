"""Apply a fitted per-source calibration (calibration.cylinder asset) to a post-log sinogram: the transferable,
object-independent parts only - detected spectrum (beam hardening), off-focal halo, room (gain-scan) scatter.
Object scatter is per fitted phantom and is NOT applied.

Per firing s (source), with t = exp(-y) the air-normalized transmission:
    t = (1 - a_s) [ (1 - g_s) P + g_s K_s(P) ] + a_s            (binned-additive model, phantom-scan scatter set equal to the
                                                                  gain-scan one: a_ph = a_g = a_s, i.e. no object scatter)
    P  -> fixed point;  L = T_s^-1(P) water mm with T_s(L) = sum_E w_s(E) exp(-mu_w(E) L),  w_s the fitted detected spectrum.
Output sinogram: water-equivalent line integrals scaled by mu_water(70 keV) (1/mm), same ray layout / mask / weights.
Sources without a calibration entry fall back to the median calibrated source.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from ...physics.xray.attenuation.mu_utils import get_mu
from ...physics.xray.correction import frame_layouts_from_views
from ...physics.xray.scatter.kernels import gaussian_matrix
from ...reconstruction import StepAndShootGeometry
from ..gpus import available_devices
from . import SKILL_VERSION, code_params, skill
from .eigen import load_sinogram

MU_W70 = 0.01928                                    # water at 70 keV, 1/mm (reference scaling of the output)


class _SpectrumLUT:
    def __init__(self, e_keV, w, device, l_max_mm=600.0, n=4096):
        e = torch.as_tensor(np.asarray(e_keV), dtype=torch.float64); w = torch.as_tensor(np.asarray(w), dtype=torch.float64).clamp_min(0); w = w / w.sum()
        mu = torch.as_tensor(get_mu("H2O", (e * 1e3).numpy(), density=1.0)).double()
        L = torch.linspace(0.0, l_max_mm, n).double()
        T = (w[None, :] * torch.exp(-L[:, None] * mu[None, :])).sum(1)
        self.Tf, self.Lf = T.flip(0).float().to(device), L.flip(0).float().to(device)

    def inverse(self, t):
        t = t.clamp(float(self.Tf[0]), float(self.Tf[-1]))
        i = torch.searchsorted(self.Tf, t.reshape(-1).contiguous()).clamp(1, len(self.Tf) - 1)
        t0, t1, l0, l1 = self.Tf[i - 1], self.Tf[i], self.Lf[i - 1], self.Lf[i]
        return (l0 + (t.reshape(-1) - t0) / (t1 - t0).clamp_min(1e-12) * (l1 - l0)).reshape(t.shape)


@skill("correction.apply_calibration")
def apply_calibration(cfg, session, job):
    """cfg: {calibration: <calibration asset id or @role>, sinogram: '@sinogram', geometry: '@geometry', role: 'sinogram_calibrated',
             use_off_focal: true, use_room_scatter: true, n_sweeps: 6, t_min: 1e-4}"""
    gid = session.resolve(cfg.get("geometry", "@geometry")); geom = StepAndShootGeometry.load(session.store.get(gid).file("geometry.pt"))
    d, sid = load_sinogram(session, cfg.get("sinogram", "@sinogram")); cid = session.resolve(cfg["calibration"])
    params = dict(use_off_focal=bool(cfg.get("use_off_focal", True)), use_room_scatter=bool(cfg.get("use_room_scatter", True)), n_sweeps=int(cfg.get("n_sweeps", 6)),
                  t_min=float(cfg.get("t_min", 1e-4)), halo_mag="calibration", code=code_params())
    inputs = {"geometry": gid, "sinogram": sid, "calibration": cid}; store = session.store; role = cfg.get("role", "sinogram_calibrated")
    a = store.lookup_recipe("sinogram", "correction.apply_calibration", SKILL_VERSION, inputs, params)
    if a is not None and not cfg.get("force"):
        print(f"apply_calibration: reused {a.id}"); return {"outputs": {role: a.id}, "metrics": {"reused": True}}
    dev = available_devices()[0]; t0 = time.time()
    cal = torch.load(store.get(cid).file("calibration.pt"), map_location="cpu", weights_only=False); src = cal["sources"]
    ids = geom.meta.get("view_source_id"); assert ids is not None, "geometry needs meta['view_source_id'] (import_corrected_step)"
    ids = torch.as_tensor(ids)
    layouts = frame_layouts_from_views(geom.S, geom.C, geom.U, geom.V, geom.pitch_u, geom.pitch_v, geom.n_u, geom.n_v, device=dev)
    # calibration entries per source id; fallback = source with median off-focal fraction
    def entry(s):
        r = src.get(s) or src.get(int(s)); return r
    fracs = {s: r["off_focal"]["fraction"] for s, r in src.items()}; med_s = sorted(fracs, key=lambda s: fracs[s])[len(fracs) // 2]
    y, w = d["y"][0].to(dev), d["w"][0].to(dev); t_all = torch.exp(-y).clamp(params["t_min"], 1.5); t_all[~w.bool()] = 1.0
    L = torch.zeros_like(y); n_fallback = 0; stats = []
    for fl in layouts:
        s = int(ids[fl.view_ids[0]]); r = entry(s)
        if r is None:
            r = src[med_s]; n_fallback += 1
        lut = _SpectrumLUT(r["spectrum"]["energies_keV"], r["spectrum"]["w_src"], dev)
        g = float(r["off_focal"]["fraction"]) if params["use_off_focal"] else 0.0
        ag = float(np.nanmean(np.asarray(r["maps"]["ag"], dtype=np.float32))) if params["use_room_scatter"] else 0.0
        t = fl.gather(t_all[None])[0]                                                   # [rows, cols]
        if g:
            mag = float(r["off_focal"].get("mag", fl.magnification))          # the calibration's own source-plane -> detector scale
            sd = math.sqrt(float(r["off_focal"]["focal_sd_mm"]) ** 2 + float(r["off_focal"]["halo_sd_mm"]) ** 2) * mag
            K = gaussian_matrix(fl.arc_mm, torch.tensor(sd, device=dev))
        P = t.clone()
        for _ in range(params["n_sweeps"]):
            inner = (t - ag) / max(1.0 - ag, 1e-6)                                      # remove room scatter (a_ph = a_g assumption)
            G = (1.0 + (P.mean(0, keepdim=True) - 1.0) @ K.T) if g else 0.0             # row-averaged halo, air outside
            P = ((inner - g * G) / (1.0 - g)).clamp(params["t_min"], 1.5)
        Lmm = lut.inverse(P); out = torch.empty_like(y); fl.scatter(Lmm[None], out[None]); L[fl.ray_slices[0][0]:fl.ray_slices[-1][1]] = out[fl.ray_slices[0][0]:fl.ray_slices[-1][1]]
        stats.append((s, g, ag))
    L = L * MU_W70 * w.float()
    m = w.bool() & (y > 0.5); ratio = float((L[m] / y[m]).median()) if m.any() else float("nan")
    st = store.stage_dir("sinogram")
    torch.save(dict(y=L[None].cpu(), w=d["w"], n_view=d["n_view"], n_u=d["n_u"], n_v=d["n_v"], B=d.get("B", 1), order=d.get("order"),
                    units=f"water-equivalent line integral x mu_water(70 keV)={MU_W70}/mm; calibration {cid} (spectrum, off-focal, room scatter; no object scatter)"), os.path.join(st, "sinogram.pt"))
    metrics = dict(t_s=round(time.time() - t0, 1), n_firings=len(layouts), n_fallback_sources=n_fallback, median_ratio_cal_over_raw_y_gt_0p5=ratio,
                   off_focal_fraction_median=float(np.median([g for _, g, _ in stats])), room_scatter_median=float(np.median([ag for _, _, ag in stats])), device=dev)
    a = store.put_computed("sinogram", st, "correction.apply_calibration", SKILL_VERSION, inputs, params, meta=dict(metrics, parent=sid, calibration=cid))
    print(f"apply_calibration -> {a.id} in {metrics['t_s']} s; {len(layouts)} firings ({n_fallback} fallback), median corrected/raw (y>0.5) {ratio:.3f}, "
          f"off-focal {metrics['off_focal_fraction_median']*100:.1f}%, room scatter {metrics['room_scatter_median']*100:.1f}%")
    return {"outputs": {role: a.id}, "metrics": metrics}
