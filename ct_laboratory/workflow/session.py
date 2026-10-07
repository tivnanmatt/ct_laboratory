"""Sessions and jobs.

A **session** is a directory (``<scan>/recon/sessions/<name>/`` by convention) holding
``session.yaml`` - a map from *roles* (``geometry``, ``sinogram``, ``eigen@64`` ...) to asset
ids - and a ``jobs/`` folder.  It is the human-facing record of what a server can do for one
scan and of what was run; it never holds data itself, only pointers into an
:class:`~ct_laboratory.workflow.assets.AssetStore`.

A **job** is one skill run inside a session: ``jobs/<id>/`` with ``job.yaml`` (the config),
``status.yaml`` (state, timings, host, GPUs, code versions, output asset ids) and ``log.txt``.
"""
from __future__ import annotations

import datetime as _dt
import os
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Callable

import yaml

from .assets import AssetStore
from .gpus import gpu_info

__all__ = ["Session", "JobRecord", "run_job", "code_versions", "code_fingerprints"]


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def code_versions(extra_repos: dict[str, str] | None = None) -> dict[str, str]:
    """{name: '<sha>[+<diffhash>] <subject>'} for ct_laboratory and any extra repo paths."""
    return {k: v["label"] for k, v in code_fingerprints(extra_repos).items()}


def code_fingerprints(extra_repos: dict[str, str] | None = None) -> dict[str, dict]:
    """Exact code identity per repo: commit sha plus, if the tree is dirty, the sha256 of
    git diff HEAD (tracked changes) - so a recipe computed with edited code is never
    confused with one computed at the clean commit.  Untracked files are not seen."""
    import hashlib
    import ct_laboratory
    repos = {"ct_laboratory": os.path.dirname(os.path.dirname(os.path.abspath(ct_laboratory.__file__)))}
    repos.update(extra_repos or {})
    out = {}
    for name, path in repos.items():
        try:
            sha = subprocess.run(["git", "-C", path, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
            subj = subprocess.run(["git", "-C", path, "log", "-1", "--format=%s"], capture_output=True, text=True, timeout=10).stdout.strip()
            diff = subprocess.run(["git", "-C", path, "diff", "HEAD"], capture_output=True, text=True, timeout=30).stdout
            fp = sha + (f"+{hashlib.sha256(diff.encode()).hexdigest()[:10]}" if diff.strip() else "")
            out[name] = {"fingerprint": fp or "?", "label": f"{fp or '?'} {subj}"}
        except Exception:
            out[name] = {"fingerprint": "?", "label": "?"}
    return out


class Session:
    def __init__(self, path: str, store: AssetStore):
        self.path = os.path.abspath(path)
        self.store = store
        self.file = os.path.join(self.path, "session.yaml")
        self.jobs_dir = os.path.join(self.path, "jobs")
        os.makedirs(self.jobs_dir, exist_ok=True)
        if not os.path.exists(self.file):
            self._write({"name": os.path.basename(self.path), "created": _now(), "roles": {}, "notes": ""})

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def data(self) -> dict:
        with open(self.file) as f:
            return yaml.safe_load(f)

    def _write(self, d: dict) -> None:
        tmp = self.file + ".tmp"
        with open(tmp, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)
        os.replace(tmp, self.file)

    @property
    def roles(self) -> dict[str, str]:
        return dict(self.data["roles"])

    def get(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(f"session {self.name} has no role {role!r}; have {sorted(self.roles)}") from None

    def set(self, role: str, asset_id: str) -> None:
        if not self.store.exists(asset_id):
            raise FileNotFoundError(f"asset {asset_id} not in store {self.store.root}")
        d = self.data; d["roles"][role] = asset_id; d["updated"] = _now(); self._write(d)

    def resolve(self, ref: str) -> str:
        """``@role`` -> asset id, otherwise the string is taken as an asset id."""
        return self.get(ref[1:]) if ref.startswith("@") else ref

    def jobs(self) -> list["JobRecord"]:
        return [JobRecord(os.path.join(self.jobs_dir, j)) for j in sorted(os.listdir(self.jobs_dir))]


class JobRecord:
    def __init__(self, path: str):
        self.path = path
        self.id = os.path.basename(path)

    @property
    def status(self) -> dict:
        p = os.path.join(self.path, "status.yaml")
        if not os.path.exists(p):
            return {"state": "UNKNOWN"}
        with open(p) as f:
            return yaml.safe_load(f)

    def write_status(self, **upd: Any) -> dict:
        s = self.status; s.update(upd)
        tmp = os.path.join(self.path, "status.yaml.tmp")
        with open(tmp, "w") as f:
            yaml.safe_dump(s, f, sort_keys=False)
        os.replace(tmp, os.path.join(self.path, "status.yaml"))
        return s


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def run_job(session: Session, skill: str, config: dict, fn: Callable[[dict, Session, "JobRecord"], dict],
            extra_repos: dict[str, str] | None = None) -> JobRecord:
    """Run ``fn(config, session, job)`` as a recorded job.  ``fn`` returns
    ``{"outputs": {role: asset_id}, "metrics": {...}}``; outputs are bound to the session roles."""
    jid = f"{_dt.datetime.now():%Y%m%d_%H%M%S}_{skill.replace('.', '_')}"
    job = JobRecord(os.path.join(session.jobs_dir, jid)); os.makedirs(job.path)
    with open(os.path.join(job.path, "job.yaml"), "w") as f:
        yaml.safe_dump({"skill": skill, "session": session.name, "config": config}, f, sort_keys=False)
    job.write_status(state="RUNNING", skill=skill, start=_now(), host=socket.gethostname(),
                     gpus=gpu_info(), code=code_versions(extra_repos), pid=os.getpid())
    t0 = time.time()
    log = open(os.path.join(job.path, "log.txt"), "a")
    out, err = sys.stdout, sys.stderr
    sys.stdout = _Tee(out, log); sys.stderr = _Tee(err, log)
    try:
        print(f"[job {jid}] {skill} in session {session.name}", flush=True)
        res = fn(config, session, job) or {}
        for role, aid in (res.get("outputs") or {}).items():
            session.set(role, aid)
        job.write_status(state="COMPLETED", end=_now(), wall_s=round(time.time() - t0, 2),
                         outputs=res.get("outputs") or {}, metrics=res.get("metrics") or {})
        print(f"[job {jid}] COMPLETED in {time.time() - t0:.1f} s; outputs {res.get('outputs')}", flush=True)
    except BaseException as e:
        traceback.print_exc()
        job.write_status(state="FAILED", end=_now(), wall_s=round(time.time() - t0, 2), error=repr(e))
        raise
    finally:
        sys.stdout, sys.stderr = out, err
        log.close()
    return job
