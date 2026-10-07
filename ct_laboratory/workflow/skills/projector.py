"""Projector assets: a scan-specific ProjectorSpec (geometry + grid + binning + window + rotation
selection) stored as an asset; eigen/recon jobs take it by role and rebuild the operator on any machine."""
from __future__ import annotations

import os

import yaml

from ...reconstruction import ProjectorSpec, StepAndShootGeometry
from . import SKILL_VERSION, code_params, skill


def load_projector(session, ref="@projector"):
    """(ProjectorSpec, geometry, projector asset id)"""
    pid = session.resolve(ref)
    spec = ProjectorSpec.from_dict(yaml.safe_load(open(session.store.get(pid).file("projector.yaml"))))
    geom = StepAndShootGeometry.load(session.store.get(spec.geometry_id).file("geometry.pt"))
    return spec, geom, pid


@skill("projector.build")
def projector_build(cfg, session, job):
    """cfg: {geometry: '@geometry', nx: 64, B: 8 (default 512//nx), dz: 2.0, fov: 512, n_win: null,
             rotations: null | [i, j, ...] | {every: 4} | {every: 4, start: 0}, z_center: null (mm, centre of a forced window), role: 'projector@64'}
    Builds the operator once here to validate it and record its size; the asset is the spec."""
    gid = session.resolve(cfg.get("geometry", "@geometry"))
    geom = StepAndShootGeometry.load(session.store.get(gid).file("geometry.pt"))
    nx = int(cfg["nx"]); rots = cfg.get("rotations")
    if isinstance(rots, dict):
        rots = list(range(int(rots.get("start", 0)), geom.n_rot, int(rots["every"])))
    spec = ProjectorSpec(geometry_id=gid, nx=nx, B=int(cfg.get("B", 512 // nx)), dz_slice=float(cfg.get("dz", 2.0)), fov=float(cfg.get("fov", 512.0)),
                         n_win=cfg.get("n_win"), rotations=rots, cache=cfg.get("cache", "column"), z_center=cfg.get("z_center"))
    store = session.store; role = cfg.get("role", f"projector@{nx}")
    params = dict(spec.to_dict(), code=code_params())
    a = store.lookup_recipe("projector", "projector.build", SKILL_VERSION, {"geometry": gid}, params)
    if a is None:
        op = spec.build(geom)
        info = op.describe()
        st = store.stage_dir("projector")
        with open(os.path.join(st, "projector.yaml"), "w") as f:
            yaml.safe_dump(dict(spec.to_dict(), built=info), f, sort_keys=False)
        a = store.put_computed("projector", st, "projector.build", SKILL_VERSION, {"geometry": gid}, params, meta=info)
        print(f"projector {role}: {info} -> {a.id}")
    else:
        print(f"projector {role}: reused {a.id}")
    return {"outputs": {role: a.id}, "metrics": a.meta.get("n_win") and dict(n_win=a.meta["n_win"], n_tot=a.meta["n_tot"], n_rot=a.meta["n_rot"], n_ray=a.meta["n_ray"], t_build_s=a.meta["t_build_s"]) or {}}
