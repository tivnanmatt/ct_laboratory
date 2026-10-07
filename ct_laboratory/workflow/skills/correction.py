"""Physics correction skill: sinogram (uncorrected post-log) -> sinogram (linearized, weighted)."""
from __future__ import annotations

import os
import time

import torch

from ...physics.xray.correction import WaterBeamHardeningLUT, frame_layouts_from_views, linearize_sinogram
from ..gpus import available_devices
from . import SKILL_VERSION, code_params, skill
from .eigen import load_geometry, load_sinogram


@skill("correction.linearize")
def correction_linearize(cfg, session, job):
    """cfg: {sinogram: '@sinogram', geometry: '@geometry', role: 'sinogram_corrected',
             spectrum: {kvp: 120, al_mm: 3.0, w_mm: 0.02}, off_focal: {fraction: 0.129, halo_sd_mm: 61.0, focal_fwhm_mm: 1.0},
             pedestal: 0.0, n_sweeps: 6, units: '1/mm', chunk_rotations: 8}
    Output: a new sinogram asset (same layout) with beam-hardening + off-focal + pedestal corrected line integrals and weights."""
    geom, gid = load_geometry(session, cfg.get("geometry", "@geometry"))
    d, sid = load_sinogram(session, cfg.get("sinogram", "@sinogram"))
    sp, of = dict(kvp=120.0, al_mm=3.0, w_mm=0.02), dict(fraction=0.129, halo_sd_mm=61.0, focal_fwhm_mm=1.0)
    sp.update(cfg.get("spectrum") or {}); of.update(cfg.get("off_focal") or {})
    params = dict(spectrum=sp, off_focal=of, pedestal=float(cfg.get("pedestal", 0.0)), n_sweeps=int(cfg.get("n_sweeps", 6)),
                  units=cfg.get("units", "1/mm"), code=code_params())
    inputs = {"geometry": gid, "sinogram": sid}
    store = session.store; role = cfg.get("role", "sinogram_corrected")
    a = store.lookup_recipe("sinogram", "correction.linearize", SKILL_VERSION, inputs, params)
    if a is not None and not cfg.get("force"):
        print(f"correction: reused {a.id}"); return {"outputs": {role: a.id}, "metrics": {"reused": True}}
    dev = available_devices()[0]
    t0 = time.time()
    lut = WaterBeamHardeningLUT(kvp=sp["kvp"], al_mm=sp["al_mm"], w_mm=sp["w_mm"], device=dev)
    fls = frame_layouts_from_views(geom.S, geom.C, geom.U, geom.V, geom.pitch_u, geom.pitch_v, geom.n_u, geom.n_v, device=dev)
    y, w = d["y"], d["w"]; R = y.shape[0]; ch = int(cfg.get("chunk_rotations", 8))
    L = torch.empty_like(y); W = torch.empty(y.shape, dtype=torch.float32)
    for a0 in range(0, R, ch):
        Lc, Wc = linearize_sinogram(y[a0:a0 + ch].to(dev), w[a0:a0 + ch].to(dev), fls, lut, of["fraction"], of["halo_sd_mm"], of["focal_fwhm_mm"],
                                    params["pedestal"], params["n_sweeps"], units=params["units"])
        L[a0:a0 + ch], W[a0:a0 + ch] = Lc.cpu(), Wc.cpu()
    m = w.bool() & (y > 0.5); ratio = float((L[m] / y[m]).median()) if m.any() else float("nan")   # attenuating rays only
    stage = store.stage_dir("sinogram")
    torch.save(dict(y=L, w=W, n_view=d["n_view"], n_u=d["n_u"], n_v=d["n_v"], B=d.get("B", 1), order=d.get("order"),
                    units=f"water-equivalent line integral ({params['units']}, mu_ref {lut.mu_ref:.5f}/mm)", weights="I0*t (relative Poisson)"),
               os.path.join(stage, "sinogram.pt"))
    metrics = dict(t_s=round(time.time() - t0, 1), mu_ref_per_mm=lut.mu_ref, n_firings=len(fls), median_ratio_corrected_over_raw_y_gt_0p5=ratio, device=dev)
    a = store.put_computed("sinogram", stage, "correction.linearize", SKILL_VERSION, inputs, params, meta=dict(metrics, parent=sid))
    print(f"correction -> {a.id} ({a.size_bytes/1e9:.2f} GB) in {metrics['t_s']} s; median corrected/raw {ratio:.3f}")
    return {"outputs": {role: a.id}, "metrics": metrics}
