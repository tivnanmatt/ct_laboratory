"""Window eigendecomposition for the rolling-window preconditioner (one asset per geometry x grid x k)."""
from __future__ import annotations

import os
import time

import torch

from ...reconstruction import RollingWindowOperator, StepAndShootGeometry, bin_sinogram, window_eigen
from ..gpus import available_devices

from . import SKILL_VERSION, code_params, skill


def load_geometry(session, ref="@geometry") -> tuple[StepAndShootGeometry, str]:
    aid = session.resolve(ref)
    return StepAndShootGeometry.load(session.store.get(aid).file("geometry.pt")), aid


def load_sinogram(session, ref="@sinogram"):
    aid = session.resolve(ref)
    d = torch.load(session.store.get(aid).file("sinogram.pt"), map_location="cpu", weights_only=False)
    return d, aid


def eigen_recipe(geom_id, sino_id, nx, B, k, fov, dz):
    inputs = {"geometry": geom_id}
    if sino_id:
        inputs["sinogram"] = sino_id          # weighted Gram: the mask enters the basis
    return inputs, dict(nx=nx, B=B, k=k, fov=fov, dz=dz, weighted=bool(sino_id), code=code_params())


def get_or_compute_eigen(session, geom, geom_id, nx, B, k, fov, dz, w1=None, sino_id=None, devices=None, log=print):
    """Return (SparseEigenDecomposition, asset_id, metrics); computes and stores the asset if missing."""
    store = session.store
    inputs, params = eigen_recipe(geom_id, sino_id, nx, B, k, fov, dz)
    a = store.lookup_recipe("eigen", "eigen.compute", SKILL_VERSION, inputs, params)
    op = RollingWindowOperator(geom, nx, B, fov, dz, devices)
    if a is not None:
        from ct_laboratory.sparse_eigen_preconditioner import SparseEigenDecomposition
        dec = SparseEigenDecomposition(gram=op.window_gram(), k=k, volume_shape=op.wshape, device=op.dev).load(a.file("eigen.pt"))
        log(f"eigen@{nx} k={k}: reused {a.id}")
        return dec, a.id, dict(reused=True, t_eig_s=0.0, cond=float(dec.eigenvalues.max() / dec.eigenvalues.min()))
    w_mean = None
    if w1 is not None:
        _, w = bin_sinogram(torch.zeros_like(w1, dtype=torch.float32), w1, geom.n_view, geom.n_u, geom.n_v, B)
        w_mean = w.to(op.dev).mean(0)
    t = time.time(); dec = window_eigen(op, k, w_mean); t_eig = time.time() - t
    lam = dec.eigenvalues
    stage = store.stage_dir("eigen"); dec.save(os.path.join(stage, "eigen.pt"))
    m = dict(reused=False, t_eig_s=round(t_eig, 2), cond=float(lam.max() / lam.min()), n_window=int(torch.tensor(op.wshape).prod()),
             devices=op.devices, t_build_s=op.t_build)
    a = store.put_computed("eigen", stage, "eigen.compute", SKILL_VERSION, inputs, params,
                           meta=dict(nx=nx, B=B, k=k, n_win=op.n_win, cond=m["cond"], eigenvalues_minmax=[float(lam.min()), float(lam.max())], t_eig_s=m["t_eig_s"]))
    log(f"eigen@{nx} k={k}: computed in {t_eig:.1f} s on {op.devices} (cond {m['cond']:.2f}) -> {a.id}")
    return dec, a.id, m


@skill("eigen.compute")
def eigen_compute(cfg, session, job):
    """cfg: {nx: 64, k: 32, B: 8 (default 512//nx), weighted: true, fov: 512, dz: 2}"""
    nx, k = int(cfg["nx"]), int(cfg["k"]); B = int(cfg.get("B", 512 // nx))
    geom, gid = load_geometry(session, cfg.get("geometry", "@geometry"))
    w1 = sid = None
    if cfg.get("weighted", True):
        d, sid = load_sinogram(session, cfg.get("sinogram", "@sinogram")); w1 = d["w"]
    _, aid, m = get_or_compute_eigen(session, geom, gid, nx, B, k, float(cfg.get("fov", 512.0)), float(cfg.get("dz", 2.0)),
                                     w1, sid, available_devices(cfg.get("max_gpus")))
    return {"outputs": {f"eigen@{nx}": aid}, "metrics": m}


@skill("eigen.sweep")
def eigen_sweep(cfg, session, job):
    """cfg: {nx: 64, ks: [32, 128, 512], ...as eigen.compute}  -> one eigen asset per k, timing + cond per k"""
    out, metrics = {}, {}
    for k in cfg["ks"]:
        r = eigen_compute(dict(cfg, k=k), session, job)
        out[f"eigen@{cfg['nx']}/k{k}"] = r["outputs"][f"eigen@{cfg['nx']}"]; metrics[f"k{k}"] = r["metrics"]
    return {"outputs": out, "metrics": metrics}
