"""Multi-resolution cascade reconstruction (all visible GPUs)."""
from __future__ import annotations

import json
import os
import time

import torch

from ...reconstruction import cascade
from ..gpus import available_devices

from . import SKILL_VERSION, code_params, skill
from .eigen import get_or_compute_eigen, load_geometry, load_sinogram


@skill("recon.cascade")
def recon_cascade(cfg, session, job):
    """cfg:
      levels: [{nx: 64, iters: 64, k: 32}, {nx: 128, iters: 64, k: 32}, ...]   (B defaults to 512//nx)
      beta_scale: 1.0   fov: 512   dz: 2   rotations: null | [a, b]  (subset of model rotations)
      max_gpus: null    save_levels: true
    Produces recon@<nx_last>: volume_<nx>.pt per level + metrics.json."""
    fov, dz = float(cfg.get("fov", 512.0)), float(cfg.get("dz", 2.0))
    geom, gid = load_geometry(session, cfg.get("geometry", "@geometry"))
    d, sid = load_sinogram(session, cfg.get("sinogram", "@sinogram"))
    y1, w1 = d["y"], d["w"]
    if cfg.get("rotations"):
        a, b = cfg["rotations"]; y1, w1 = y1[a:b + 1], w1[a:b + 1]
        geom.n_rot = b - a + 1
    levels = [dict(nx=int(l["nx"]), B=int(l.get("B", 512 // int(l["nx"]))), iters=int(l["iters"]), k=int(l.get("k", 32)),
                   beta_scale=float(l.get("beta_scale", cfg.get("beta_scale", 1.0)))) for l in cfg["levels"]]
    devices = available_devices(cfg.get("max_gpus"))
    inputs = {"geometry": gid, "sinogram": sid}
    params = dict(levels=levels, fov=fov, dz=dz, rotations=cfg.get("rotations"), code=code_params())
    store = session.store
    existing = store.lookup_recipe("recon", "recon.cascade", SKILL_VERSION, inputs, params)
    role = f"recon@{levels[-1]['nx']}"
    if existing is not None and not cfg.get("force", False):
        print(f"recon: reused {existing.id}")
        return {"outputs": {role: existing.id}, "metrics": dict(reused=True, **json.load(open(existing.file("metrics.json"))))}
    eig_used = {}

    def provider(op, k, w_mean):
        dec, aid, _ = get_or_compute_eigen(session, geom, gid, op.nx, op.B, k, fov, dz, w1 if cfg.get("weighted_eigen", True) else None,
                                           sid if cfg.get("weighted_eigen", True) else None, devices)
        eig_used[f"eigen@{op.nx}"] = aid
        return dec

    t0 = time.time()
    vols, metrics = cascade(geom, y1, w1, levels, provider, fov, dz, devices, log=print)
    stage = store.stage_dir("recon")
    for nx, v in vols:
        torch.save(v, os.path.join(stage, f"volume_{nx}.pt"))
    summary = dict(levels=metrics, t_total_s=round(time.time() - t0, 2), devices=devices, eigen=eig_used, n_rot=geom.n_rot)
    json.dump(summary, open(os.path.join(stage, "metrics.json"), "w"), indent=1)
    a = store.put_computed("recon", stage, "recon.cascade", SKILL_VERSION, inputs, params,
                           meta=dict(levels=[l["nx"] for l in levels], t_total_s=summary["t_total_s"], eigen=eig_used, devices=devices))
    print(f"recon -> {a.id} ({a.size_bytes/1e6:.0f} MB) in {summary['t_total_s']} s on {len(devices)} GPU(s)")
    return {"outputs": {role: a.id, **eig_used}, "metrics": summary}
