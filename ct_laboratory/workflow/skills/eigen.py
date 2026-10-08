"""Window eigendecomposition for the rolling-window preconditioner: one asset per (projector, k, weights)."""
from __future__ import annotations

import os
import time

import torch

from ...reconstruction import bin_sinogram, window_eigen
from ...sparse_eigen_preconditioner import SparseEigenDecomposition
from ..gpus import available_devices
from . import SKILL_VERSION, code_params, skill
from .projector import load_projector


def load_sinogram(session, ref="@sinogram"):
    aid = session.resolve(ref)
    return torch.load(session.store.get(aid).file("sinogram.pt"), map_location="cpu", weights_only=False), aid


def get_or_compute_eigen(session, op, pid, k, w1=None, sino_id=None, log=print, method="eigsh"):
    """(SparseEigenDecomposition, asset id, metrics) for projector asset ``pid``; computes and stores if missing."""
    store = session.store
    inputs = {"projector": pid}
    if sino_id:
        inputs["sinogram"] = sino_id          # weighted Gram: the mask enters the basis
    params = dict(k=k, weighted=bool(sino_id), code=code_params(), **({} if method == "eigsh" else {"method": method}))
    a = store.lookup_recipe("eigen", "eigen.compute", SKILL_VERSION, inputs, params)
    if a is not None:
        dec = SparseEigenDecomposition(gram=op.window_gram(), k=k, volume_shape=op.wshape, device=op.dev).load(a.file("eigen.pt"))
        log(f"eigen k={k} for {pid}: reused {a.id}")
        return dec, a.id, dict(reused=True, t_eig_s=0.0, cond=float(dec.eigenvalues.max() / dec.eigenvalues.min()))
    w_mean = None
    if w1 is not None:
        g = op.geom
        _, w = bin_sinogram(torch.zeros(w1.shape, dtype=torch.float32), w1, g.n_view, g.n_u, g.n_v, op.B)
        w_mean = w.to(op.dev).mean(0)
    t = time.time(); dec = window_eigen(op, k, w_mean, method=method); t_eig = time.time() - t
    lam = dec.eigenvalues
    st = store.stage_dir("eigen"); dec.save(os.path.join(st, "eigen.pt"))
    m = dict(reused=False, method=method, **getattr(dec, "last_solver_stats", {}), t_eig_s=round(t_eig, 2), cond=float(lam.max() / lam.min()), n_window=int(torch.tensor(op.wshape).prod()), devices=op.devices)
    a = store.put_computed("eigen", st, "eigen.compute", SKILL_VERSION, inputs, params,
                           meta=dict(method=method, k=k, nx=op.nx, B=op.B, n_win=op.n_win, cond=m["cond"], eigenvalues_minmax=[float(lam.min()), float(lam.max())], t_eig_s=m["t_eig_s"]))
    log(f"eigen k={k} for {pid}: computed in {t_eig:.1f} s on {op.devices} (cond {m['cond']:.2f}) -> {a.id}")
    return dec, a.id, m


@skill("eigen.compute")
def eigen_compute(cfg, session, job):
    """cfg: {projector: '@projector@64', k: 32, weighted: true, sinogram: '@sinogram', role: 'eigen@64', method: eigsh | cupy_eigsh}"""
    spec, geom, pid = load_projector(session, cfg.get("projector", "@projector"))
    op = spec.build(geom, available_devices(cfg.get("max_gpus")))
    w1 = sid = None
    if cfg.get("weighted", True):
        d, sid = load_sinogram(session, cfg.get("sinogram", "@sinogram")); w1 = d["w"][op.rotations]
    _, aid, m = get_or_compute_eigen(session, op, pid, int(cfg["k"]), w1, sid, method=cfg.get("method", "eigsh"))
    return {"outputs": {cfg.get("role", f"eigen@{op.nx}"): aid}, "metrics": m}


@skill("eigen.sweep")
def eigen_sweep(cfg, session, job):
    """cfg: {projector: '@projector@64', ks: [32, 128, 512], ...}  -> eigen@<nx>/k<k> per k"""
    out, metrics = {}, {}
    spec, _, _ = load_projector(session, cfg.get("projector", "@projector"))
    for k in cfg["ks"]:
        r = eigen_compute(dict(cfg, k=k, role=f"eigen@{spec.nx}/k{k}"), session, job)
        out.update(r["outputs"]); metrics[f"k{k}"] = r["metrics"]
    return {"outputs": out, "metrics": metrics}
