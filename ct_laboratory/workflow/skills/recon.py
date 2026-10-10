"""Multi-resolution cascade reconstruction: a list of projector assets (coarse to fine) + a sinogram."""
from __future__ import annotations

import json
import os
import time

import torch

from ...reconstruction import cascade
from ..gpus import available_devices
from . import SKILL_VERSION, code_params, skill
from .eigen import get_or_compute_eigen, load_sinogram
from .projector import load_projector


@skill("recon.cascade")
def recon_cascade(cfg, session, job):
    """cfg:
      sinogram: '@sinogram'
      levels: [{projector: '@projector@64', iters: 64, k: 32}, {projector: '@projector@128', iters: 48, k: 32}, ...]
      beta_scale: 1.0   scaling: sensitivity | none (per level too)   weighted_eigen: true   eigen_method: eigsh | cupy_eigsh   max_gpus: null   role: 'recon@<nx_last>'
    Levels may select different station subsets (e.g. every 4th at 8 mm); the warm start is resampled in 3-D.  Produces volume_<nx>.pt per level + metrics.json."""
    d, sid = load_sinogram(session, cfg.get("sinogram", "@sinogram"))
    devices = available_devices(cfg.get("max_gpus"))
    ops, pids = [], []
    for l in cfg["levels"]:
        spec, geom, pid = load_projector(session, l["projector"]); ops.append(spec.build(geom, devices)); pids.append(pid)
    y1, w1 = d["y"], d["w"]                                   # all stations; each level selects its own rotations (cascade())
    levels = [dict(op=op, iters=int(l["iters"]), k=int(l.get("k", 32)), beta_scale=float(l.get("beta_scale", cfg.get("beta_scale", 1.0))),
                   scaling=l.get("scaling", cfg.get("scaling", "sensitivity"))) for op, l in zip(ops, cfg["levels"])]
    inputs = {"sinogram": sid, **{f"projector@{op.nx}": pid for op, pid in zip(ops, pids)}}
    params = dict(levels=[dict(projector=pid, iters=lv["iters"], k=lv["k"], beta_scale=lv["beta_scale"], scaling=lv["scaling"]) for pid, lv in zip(pids, levels)],
                  weighted_eigen=bool(cfg.get("weighted_eigen", True)), code=code_params(),
                  **({} if cfg.get("eigen_method", "eigsh") == "eigsh" else {"eigen_method": cfg["eigen_method"]}))
    store = session.store; role = cfg.get("role", f"recon@{ops[-1].nx}")
    existing = store.lookup_recipe("recon", "recon.cascade", SKILL_VERSION, inputs, params)
    if existing is not None and not cfg.get("force", False):
        print(f"recon: reused {existing.id}")
        return {"outputs": {role: existing.id}, "metrics": dict(reused=True, **json.load(open(existing.file("metrics.json"))))}
    eig_used = {}

    def provider(op, k, w_mean, beta=0.0, scale=None):
        pid = pids[ops.index(op)]
        dec, aid, _ = get_or_compute_eigen(session, op, pid, k, w1 if params["weighted_eigen"] else None, sid if params["weighted_eigen"] else None, method=cfg.get("eigen_method", "eigsh"),
                                           beta=float(beta), scaling="sensitivity" if scale is not None else "none")
        eig_used[f"eigen@{op.nx}"] = aid
        return dec

    t0 = time.time()
    vols, metrics = cascade(y1, w1, levels, provider, log=print)
    st = store.stage_dir("recon")
    for nx, v in vols:
        torch.save(v, os.path.join(st, f"volume_{nx}.pt"))
    summary = dict(levels=metrics, t_total_s=round(time.time() - t0, 2), devices=devices, eigen=eig_used, rotations=rots, projectors=pids)
    json.dump(summary, open(os.path.join(st, "metrics.json"), "w"), indent=1)
    a = store.put_computed("recon", st, "recon.cascade", SKILL_VERSION, inputs, params,
                           meta=dict(levels=[op.nx for op in ops], t_total_s=summary["t_total_s"], eigen=eig_used, devices=devices))
    print(f"recon -> {a.id} ({a.size_bytes/1e6:.0f} MB) in {summary['t_total_s']} s on {len(devices)} GPU(s)")
    return {"outputs": {role: a.id, **eig_used}, "metrics": summary}
