"""Calibration skill: per-source X-ray system calibration on a known cylinder phantom (physics.xray.calibration.CylinderCalibration).

Input assets
  geometry      StepAndShootGeometry
  rawsinogram   rawsinogram.pt (bin 1, same ray order as the geometry views): y [n_rot, n_ray] photon-equivalent counts (shifted
                Poisson), I0 [n_ray] air signal, kappa [n_ray], good [n_ray] bool, glitch [n_rot, n_ray] bool (optional),
                extra_L [n_rot, n_ray] fixed water-equivalent path (optional, e.g. table), fluence [n_ray] smooth air fluence (optional)
Config
  rotations:    [first, last] model rotations whose rays cross the uniform section (one firing per source per rotation)
  phantom:      {radius_mm, cx, cy, density, material: water | {formula, density}, z_slab: [lo, hi], radius_sd, density_sd, axis_sd}
  spectrum:     {kvp, al_mm, w_mm, heel, spline, csi_mm, spek: path to spek npz or null (spekpy)}; off_focal {fraction, halo_sd_mm}
  room_fraction, floor, jitter_px, eps_sigma, laplacian_gain, laplacian_phantom, steps, lr, sources: [ids] or null (all)
Output
  calibration asset: calibration.pt = {sources: {sid: result}, energies_keV, config}; one entry per fitted source with parameters,
  per-pixel maps (a_g, a_ph, eps in the firing's ray order), detected spectrum and the corrected-data length test.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch

from ...physics.xray.calibration import CylinderPhantom, FiringData, CylinderCalibrationConfig, CylinderCalibration
from ...physics.xray.correction import frame_layouts_from_views
from ..gpus import available_devices
from . import SKILL_VERSION, code_params, skill


def _mu(material, keV):
    import xraydb
    if isinstance(material, str):
        return xraydb.material_mu(material, keV) / 10
    d = xraydb.chemparse(material["formula"]); tot = sum(n * xraydb.atomic_mass(e) for e, n in d.items())
    return material["density"] * sum(n * xraydb.atomic_mass(e) / tot * xraydb.mu_elam(e, keV) for e, n in d.items()) / 10


def _spek(cfg, edges):
    sp = cfg.get("spek")
    if sp:
        z = np.load(sp); phi = z["phi"]; kvs = z["kvs"]
        if phi.shape[1] == 2 * (len(edges) - 1): phi = phi.reshape(len(kvs), -1, 2).sum(2)
        return kvs, phi
    from ...physics.xray.source import spekpy_table
    kvp = float(cfg.get("kvp", 120.0)); kvs = np.arange(kvp - 10, kvp + 11, 2.0)
    return kvs, spekpy_table(kvs, edges)


@skill("calibration.cylinder")
def calibration_cylinder(cfg, session, job):
    import xraydb
    from ...reconstruction import StepAndShootGeometry
    gid = session.resolve(cfg.get("geometry", "@geometry")); geom = StepAndShootGeometry.load(session.store.get(gid).file("geometry.pt"))
    rid = session.resolve(cfg.get("rawsinogram", "@rawsinogram")); raw = torch.load(session.store.get(rid).file("rawsinogram.pt"), map_location="cpu", weights_only=False, mmap=True)
    store = session.store; role = cfg.get("role", "calibration")
    params = {k: cfg[k] for k in cfg if k not in ("geometry", "rawsinogram", "role", "force")}; params["code"] = code_params()
    inputs = {"geometry": gid, "rawsinogram": rid}
    a = store.lookup_recipe("calibration", "calibration.cylinder", SKILL_VERSION, inputs, params)
    if a is not None and not cfg.get("force"):
        print(f"calibration: reused {a.id}"); return {"outputs": {role: a.id}, "metrics": {"reused": True}}
    dev = available_devices()[0]; t0 = time.time()
    r0, r1 = cfg["rotations"]; rots = list(range(int(r0), int(r1) + 1)); NK = len(rots)
    edges = np.arange(10.0, 137.0, 2.0); E = 0.5 * (edges[1:] + edges[:-1]); keV = E * 1e3
    spc = dict(kvp=120.0, al_mm=17.0, w_mm=0.013, heel=0.0, spline=[0.0] * 4, csi_mm=0.6); spc.update(cfg.get("spectrum") or {})
    ofc = dict(fraction=0.129, halo_sd_mm=61.0); ofc.update(cfg.get("off_focal") or {})
    kvs, phi = _spek(spc, edges)
    csi = (0.5115 * xraydb.mu_elam("Cs", keV) + 0.4885 * xraydb.mu_elam("I", keV)) * 4.51 / 10
    ccfg = CylinderCalibrationConfig(energies_keV=E, spek_kvps=kvs, spek_phi=phi, mu_al=xraydb.material_mu("Al", keV) / 10, mu_w=xraydb.mu_elam("W", keV) * 19.3 / 10, mu_scint=csi,
                                     kvp=spc["kvp"], al_mm=spc["al_mm"], w_mm=spc["w_mm"], heel=spc["heel"], spline=list(spc["spline"]), csi_mm=spc["csi_mm"],
                                     off_focal_fraction=ofc["fraction"], halo_sd_mm=ofc["halo_sd_mm"],
                                     room_fraction=float(cfg.get("room_fraction", 0.035)), floor=float(cfg.get("floor", 0.002)), jitter_px=float(cfg.get("jitter_px", 0.195)),
                                     eps_sigma=float(cfg.get("eps_sigma", 0.01)), laplacian_gain=float(cfg.get("laplacian_gain", 5e7)), laplacian_phantom=float(cfg.get("laplacian_phantom", 1.25e6)),
                                     steps=int(cfg.get("steps", 200)), lr=float(cfg.get("lr", 0.05)))
    phc = cfg["phantom"]; mu_obj = torch.tensor(_mu(phc.get("material", "water"), keV), dtype=torch.float32)
    phantom = CylinderPhantom(radius=float(phc["radius_mm"]), cx=float(phc["cx"]), cy=float(phc["cy"]), mu=mu_obj, density=float(phc.get("density", 1.0)),
                              z_slab=tuple(phc.get("z_slab", (-1e4, 1e4))), radius_sd=float(phc.get("radius_sd", 1.0)), density_sd=float(phc.get("density_sd", 0.02)), axis_sd=float(phc.get("axis_sd", 0.5)))
    # firings: group the geometry's views by source
    layouts = frame_layouts_from_views(geom.S, geom.C, geom.U, geom.V, geom.pitch_u, geom.pitch_v, geom.n_u, geom.n_v, device=dev)
    S, C, U, V = (t.to(dev) for t in (geom.S, geom.C, geom.U, geom.V)); nu, nv = geom.n_u, geom.n_v
    Vn = V - (V * U).sum(-1, keepdim=True) * U; Vn = Vn / Vn.norm(dim=-1, keepdim=True); Nn = torch.linalg.cross(U, Vn); Nn = Nn / Nn.norm(dim=-1, keepdim=True)
    iu = torch.arange(nu, device=dev) - (nu - 1) / 2; iv = torch.arange(nv, device=dev) - (nv - 1) / 2
    NSUB = int(cfg.get("sub_rays", 3)); osub = (torch.arange(NSUB, device=dev) + 0.5) / NSUB - 0.5
    # object z per model rotation r: the volume slab origin z0 = dz_rot * (floor(zmin / dz_rot) - 1) with zmin the cone's z extent (rolling-window convention)
    zmin, _ = geom.ray_z_span(1, float(cfg.get("fov_radius_mm", 256.0)))
    z0 = float(cfg.get("z0_mm", geom.dz_rot * (math.floor(zmin / geom.dz_rot) - 1)))
    print(f"object z origin {z0:.1f} mm (rotation {rots[0]} -> slab offset {rots[0] * geom.dz_rot - z0:.1f} mm); phantom z slab {phantom.z_slab}")
    # layouts come in unique-source-key order; the raw asset's source_ids are in FIRING order (ascending ray offset) -> map by first ray offset
    by_off = sorted(range(len(layouts)), key=lambda i: layouts[i].ray_slices[0][0])
    sid_of = {by_off[j]: (j if raw.get("source_ids") is None else int(raw["source_ids"][j])) for j in range(len(by_off))}
    want = cfg.get("sources"); results = {}; n_fit = 0
    R_meas = raw["y"].shape[0]; flip = bool(cfg.get("flip_rotations", False))                     # set if the raw asset is in measured (not model) rotation order
    for li, fl in enumerate(layouts):
        sid = sid_of[li]
        if want is not None and sid not in want: continue
        ids = fl.view_ids.to(dev); nm = len(ids); npx = nm * nu * nv
        if nm == 0: continue
        sl = [slice(a_, b_) for a_, b_ in fl.ray_slices]
        d_so = float((S[ids[0]][:2]).norm()); d_sd = float((C[ids] - S[ids[0]]).norm(dim=1).mean()); mag_det = (d_sd - d_so) / d_so    # source-plane -> detector scale through isocentre
        pix = C[ids][:, None, None, :] + (iu[None, :, None, None] * geom.pitch_u) * U[ids][:, None, None, :] + (iv[None, None, :, None] * geom.pitch_v) * Vn[ids][:, None, None, :]
        sub = pix[..., None, :] + (osub[None, None, None, :, None] * geom.pitch_u) * U[ids][:, None, None, None, :]
        arc = fl.arc_mm.reshape(nm, nu); colpos = torch.argsort(torch.argsort(arc.reshape(-1))).to(dev); NC = nm * nu
        mo_ = torch.arange(nm, device=dev)[:, None, None].expand(nm, nu, nv).reshape(-1); co_ = torch.arange(nu, device=dev)[None, :, None].expand(nm, nu, nv).reshape(-1)
        ro_ = torch.arange(nv, device=dev)[None, None, :].expand(nm, nu, nv).reshape(-1); ci_ = colpos.reshape(nm, nu)[:, :, None].expand(nm, nu, nv).reshape(-1)
        def img(flat, fill=0.0):
            out = torch.full((nv, NC) + tuple(flat.shape[1:]), fill, device=dev, dtype=flat.dtype); out[ro_, ci_] = flat; return out
        gat = lambda vec: torch.cat([vec[s_] for s_ in sl])
        ks = [(R_meas - 1 - r) if flip else r for r in rots]
        Y = torch.stack([img(torch.nan_to_num(gat(raw["y"][k]).float()).to(dev)) for k in ks]); I0 = img(gat(raw["I0"]).float().to(dev)); kap = img(gat(raw["kappa"]).float().to(dev))
        good = torch.stack([img((gat(raw["good"]) & (~gat(raw["glitch"][k]) if "glitch" in raw else True)).to(dev), False) for k in ks]) & (I0 > 100)[None] & (Y < 2 * I0[None] + 1000)
        ROW = img(ro_); good &= ~((ROW == 0) | (ROW == nv - 1))[None]; I0 = I0.clamp(min=1.0)
        t = torch.nan_to_num((Y - kap[None]) / I0[None], nan=0.0, posinf=0.0, neginf=0.0)
        t = torch.where(good, t, torch.zeros_like(t)).clamp(-1.0, 3.0)                        # invalid samples (fp16 overflow, hot pixels) must not leak into the edge-gradient / variance terms
        extra = torch.stack([img(gat(raw["extra_L"][k]).float().to(dev)) for k in ks]) if "extra_L" in raw else None
        fr = None
        if "fluence" in raw:
            F = img(gat(raw["fluence"]).float().to(dev)); fr = torch.where(F > 0, F[F > 0].median() / F.clamp(min=1e-6), torch.ones_like(F))
        arcs = torch.sort(arc.reshape(-1)).values; arcs = arcs - arcs[0]
        data = FiringData(t=t, I0=I0, kappa=kap, good=good, det_pos=img(pix.reshape(-1, 3)), det_normal=img(Nn[ids][mo_]), src=S[ids[0]], sub_pos=img(sub.reshape(npx, NSUB, 3)),
                          arc_mm=arcs, row_mm=iv * geom.pitch_v, module=img(mo_), col=img(co_), row=ROW,
                          z_offsets=torch.tensor([r * geom.dz_rot for r in rots], device=dev) - z0, extra_L=extra, fluence_ratio=fr,
                          magnification=float(cfg.get("magnification", mag_det)), cone_offset_deg=float(cfg.get("cone_offset_deg", 0.0)))
        if int(good.sum()) < 1000:
            print(f"source {sid}: {int(good.sum())} good samples, skipped"); continue
        cal = CylinderCalibration(ccfg, phantom, data, device=dev)
        n_in = int((good & cal.inside).sum())
        if n_in < 1000:
            print(f"source {sid}: only {n_in} good samples inside the phantom slab, skipped"); continue
        cal.fit(log=lambda *a: None); r = cal.result()
        if not math.isfinite(r["bump"]) or not all(math.isfinite(v) for v in r["geo"]):
            print(f"source {sid}: fit diverged (NaN), skipped"); continue
        r["ray_slices"] = fl.ray_slices; r["layout"] = dict(row=ro_.cpu(), col=ci_.cpu())
        results[sid] = r; n_fit += 1
        print(f"source {sid:4d}: bump {r['bump']:+6.2f} ± {r['bump_se']:.2f} mm, floor {r['params'].get('noise floor (% of air)', float('nan')):.3f} %, "
              f"Al {r['params'].get('Al filtration (mm)', float('nan')):.1f} mm, coverage {r['coverage_1sd']:.2f} [{time.time()-t0:.0f}s]", flush=True)
    bumps = np.array([r["bump"] for r in results.values()])
    stage = store.stage_dir("calibration")
    torch.save(dict(sources=results, energies_keV=E, config=params, rotations=rots), os.path.join(stage, "calibration.pt"))
    metrics = dict(t_s=round(time.time() - t0, 1), n_sources=n_fit, bump_median_mm=float(np.median(bumps)) if n_fit else None,
                   bump_iqr_mm=[float(np.percentile(bumps, 25)), float(np.percentile(bumps, 75))] if n_fit else None, device=dev)
    a = store.put_computed("calibration", stage, "calibration.cylinder", SKILL_VERSION, inputs, params, meta=metrics)
    print(f"calibration -> {a.id}: {n_fit} sources, bump median {metrics['bump_median_mm']} mm in {metrics['t_s']} s")
    return {"outputs": {role: a.id}, "metrics": metrics}
