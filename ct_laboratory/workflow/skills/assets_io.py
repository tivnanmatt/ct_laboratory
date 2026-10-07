"""Import / export skills: get user-provided assets into the store and bind them to roles."""
from __future__ import annotations

import os

from . import skill


@skill("asset.import")
def asset_import(cfg, session, job):
    """cfg: {type: geometry|sinogram|eigen|recon, files: [paths], role: <session role>, meta: {...}}"""
    a = session.store.put_upload(cfg["type"], cfg["files"], meta=cfg.get("meta"))
    print(f"imported {a.id}: {a.files} ({a.size_bytes/1e6:.1f} MB)")
    return {"outputs": {cfg.get("role", cfg["type"]): a.id}, "metrics": {"size_bytes": a.size_bytes}}


@skill("asset.export")
def asset_export(cfg, session, job):
    """cfg: {asset: '@role' or id, dest: directory}  -> copies the asset's files out of the store"""
    import shutil
    a = session.store.get(session.resolve(cfg["asset"]))
    os.makedirs(cfg["dest"], exist_ok=True)
    for f in a.files + ["asset.yaml"]:
        shutil.copy2(a.file(f), os.path.join(cfg["dest"], f))
    print(f"exported {a.id} -> {cfg['dest']}")
    return {"metrics": {"files": len(a.files)}}
