#!/usr/bin/env python3
"""ctlab - ct_laboratory recon server/client command line (console script `ctlab`, or `python -m ct_laboratory.workflow.cli`).

  ctlab session new  <session_dir>                      create a session (any directory; <scan>/recon/sessions/<name> by convention)
  ctlab session show <session_dir>                      roles + jobs
  ctlab run  <skill> <job.yaml> -s <session_dir>        run a skill locally (all visible GPUs)
  ctlab run  <skill> <job.yaml> -s <session_dir> --remote <host> [--remote-root R] [--remote-code C]
                                                      push the job's input assets to the server, run there, pull results
  ctlab assets [-t type]                                list the store
  ctlab skills                                          list skills

The asset store root is --store or $CTLAB_STORE (default ~/recon_assets).  On a server the
same command runs with --store pointing at its own store; a session on the server is a scratch
session (its roles are only there so the skill can resolve @geometry etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

import yaml

from .assets import AssetStore
from .remote import RemoteStore
from .session import Session, run_job
from .skills import CODE_REPOS, SKILLS, register_code_repo

DEFAULT_STORE = os.environ.get("CTLAB_STORE", os.path.expanduser("~/recon_assets"))


def _refs_in(cfg: dict, session: Session) -> list[str]:
    """Asset ids referenced by a config: every '@role' string plus explicit ids, plus the session's geometry/sinogram."""
    ids = set()
    for role in ("geometry", "sinogram"):
        if role in session.roles:
            ids.add(session.get(role))

    def walk(v):
        if isinstance(v, str):
            if v.startswith("@") and v[1:] in session.roles:
                ids.add(session.get(v[1:]))
            elif "-" in v and session.store.exists(v):
                ids.add(v)
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    walk(cfg)
    return sorted(ids)


def cmd_run(a):
    store = AssetStore(a.store); session = Session(a.session, store)
    cfg = yaml.safe_load(open(a.config)) or {}
    if a.skill not in SKILLS:
        sys.exit(f"unknown skill {a.skill}; have {sorted(SKILLS)}")
    if not a.remote:
        job = run_job(session, a.skill, cfg, SKILLS[a.skill], extra_repos=CODE_REPOS)
        print(yaml.safe_dump({"job": job.id, "status": job.status["state"], "outputs": job.status.get("outputs"),
                              "wall_s": job.status.get("wall_s")}, sort_keys=False))
        return
    # ---- remote: push inputs, run the same command on the server, pull outputs, record locally
    remote = RemoteStore(a.remote, a.remote_root)
    ids = _refs_in(cfg, session)
    import time; t = time.time()
    pushed = remote.push(store, ids)
    print(f"pushed {len(pushed)} of {len(ids)} input assets in {time.time() - t:.1f} s {pushed}")
    rsess = f"{a.remote_root}/.sessions/{session.name}"
    roles = " ".join(f"{shlex.quote(r)}={shlex.quote(i)}" for r, i in session.roles.items() if i in ids)
    rcfg = shlex.quote(yaml.safe_dump(cfg))
    code = a.remote_code
    ctlab = f"python -m ct_laboratory.workflow.cli --store {shlex.quote(a.remote_root)}"
    cmd = (f"cd {shlex.quote(code)} && mkdir -p {shlex.quote(rsess)} && "
           f"{ctlab} session new {shlex.quote(rsess)} >/dev/null && "
           f"{ctlab} session bind {shlex.quote(rsess)} {roles} >/dev/null && "
           f"echo {rcfg} > /tmp/sct_job.yaml && "
           f"{ctlab} run {shlex.quote(a.skill)} /tmp/sct_job.yaml -s {shlex.quote(rsess)} --json")
    print(f"running {a.skill} on {a.remote} ...", flush=True)
    r = remote.run(cmd, check=False, capture=True, timeout=None)     # the job itself may run for hours
    sys.stdout.write(r.stdout[-4000:] if len(r.stdout) > 4000 else r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:]); sys.exit(f"remote job failed (rc {r.returncode})")
    res = json.loads(r.stdout.strip().splitlines()[-1])
    t = time.time(); pulled = remote.pull(store, list(res["outputs"].values()))
    print(f"pulled {len(pulled)} output assets in {time.time() - t:.1f} s")
    # record the remote run as a local job (bind outputs to this session)
    def done(cfg_, sess_, job_):
        print(f"remote job {res['job']} on {a.remote}: {res['status']}"); return {"outputs": res["outputs"], "metrics": dict(res.get("metrics") or {}, remote=a.remote, remote_job=res["job"], pushed_s=pushed, pulled_s=pulled)}
    job = run_job(session, a.skill, dict(cfg, _remote=a.remote), done, extra_repos=CODE_REPOS)
    print(yaml.safe_dump({"job": job.id, "status": job.status["state"], "outputs": job.status.get("outputs")}, sort_keys=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ctlab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("session"); ss = s.add_subparsers(dest="sub", required=True)
    p = ss.add_parser("new"); p.add_argument("session")
    p = ss.add_parser("show"); p.add_argument("session")
    p = ss.add_parser("bind"); p.add_argument("session"); p.add_argument("pairs", nargs="*", help="role=asset_id")
    p = sp.add_parser("run"); p.add_argument("skill"); p.add_argument("config"); p.add_argument("-s", "--session", required=True)
    p.add_argument("--remote"); p.add_argument("--remote-root", default="/root/recon_assets"); p.add_argument("--remote-code", default="/root/ct_laboratory", help="directory to cd into on the server (python must import ct_laboratory there)")
    p.add_argument("--code-repo", action="append", default=[], help="name=path of a client repo whose commit enters recipe hashes")
    p.add_argument("--json", action="store_true", help="print a one-line JSON result last (used by --remote)")
    p = sp.add_parser("assets"); p.add_argument("-t", "--type")
    sp.add_parser("skills")
    a = ap.parse_args(argv)
    for cr in a.__dict__.get("code_repo", []) or []:
        n, p_ = cr.split("=", 1); register_code_repo(n, p_)
    if a.cmd == "skills":
        for k, f in sorted(SKILLS.items()):
            print(f"{k:16} {(f.__doc__ or '').strip().splitlines()[0]}")
    elif a.cmd == "assets":
        for x in AssetStore(a.store).find(a.type):
            print(f"{x.id:28} {x.size_bytes/1e6:9.1f} MB  {x.meta.get('kind')}  {x.meta.get('skill', '')} {x.meta.get('created', '')}")
    elif a.cmd == "session":
        sess = Session(a.session, AssetStore(a.store))
        if a.sub == "bind":
            for pr in a.pairs:
                r, i = pr.split("=", 1); sess.set(r, i)
        if a.sub in ("new", "bind"):
            print(f"session {sess.name} at {sess.path}: {sess.roles}")
        else:
            print(yaml.safe_dump(sess.data, sort_keys=False))
            for j in sess.jobs():
                st = j.status; print(f"  {j.id:40} {st.get('state'):10} {st.get('wall_s', '')!s:>8} s  {st.get('outputs', '')}")
    elif a.cmd == "run":
        if a.json:
            store = AssetStore(a.store); session = Session(a.session, store)
            cfg = yaml.safe_load(open(a.config)) or {}
            job = run_job(session, a.skill, cfg, SKILLS[a.skill], extra_repos=CODE_REPOS)
            print(json.dumps({"job": job.id, "status": job.status["state"], "outputs": job.status.get("outputs") or {}, "metrics": job.status.get("metrics")}, default=str))
        else:
            cmd_run(a)


if __name__ == "__main__":
    main()
