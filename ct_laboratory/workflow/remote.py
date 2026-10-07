"""Client <-> server transport over ssh + rsync.

The server is just another machine with an :class:`AssetStore` at some root.  Because asset
ids are content/recipe hashes, syncing is "send the asset directories the other side does
not have" - no job bookkeeping crosses the wire except the one job config.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time

__all__ = ["RemoteStore"]


class RemoteStore:
    def __init__(self, host: str, root: str, ssh_opts: tuple[str, ...] = ()):
        """``host`` is an ssh alias or ``user@ip``; ``root`` the server's asset-store root."""
        self.host, self.root = host, root.rstrip("/")
        extra = tuple(os.environ.get("SCT_SSH_OPTS", "").split())   # e.g. "-F /dev_ws/.ssh/config" inside a container
        self.ssh = ["ssh", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4", "-o", "ConnectTimeout=30", *extra, *ssh_opts, host]

    RETRIES, BACKOFF_S = 4, 10

    def run(self, cmd: str, check: bool = True, capture: bool = False, timeout: float | None = 120.0) -> subprocess.CompletedProcess:
        """ssh and run cmd; an ssh-level failure (rc 255: dropped connection) is retried with backoff."""
        for attempt in range(self.RETRIES):
            try:
                r = subprocess.run([*self.ssh, cmd], text=True, capture_output=capture, timeout=timeout)
            except subprocess.TimeoutExpired:            # hung session (happens on overloaded hosts): treat like a drop
                r = subprocess.CompletedProcess([*self.ssh, cmd], 255, "", "ssh timeout")
            if r.returncode != 255:
                break
            time.sleep(self.BACKOFF_S * (attempt + 1))
        if check and r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
        return r

    _HAS_PY = ("import os,sys,yaml\n"
               "root=sys.argv[1]\n"
               "for a in sys.argv[2:]:\n"
               "    d=os.path.join(root,a.rsplit('-',1)[0],a); ok=0\n"
               "    try:\n"
               "        want=yaml.safe_load(open(os.path.join(d,'asset.yaml')))['size_bytes']\n"
               "        got=sum(os.path.getsize(os.path.join(d,f)) for f in os.listdir(d) if f!='asset.yaml')\n"
               "        ok=int(want==got)\n"
               "    except Exception: pass\n"
               "    print(a,ok)\n")

    def has(self, asset_ids: list[str]) -> dict[str, bool]:
        """Which assets the server holds COMPLETELY: asset.yaml present and the data files' total size equal to
        the size_bytes recorded in it (a partial copy from a dropped transfer counts as missing)."""
        cmd = "python3 -c " + shlex.quote(self._HAS_PY) + " " + shlex.quote(self.root) + " " + " ".join(shlex.quote(a) for a in asset_ids)
        r = self.run(cmd, capture=True)
        return {l.split()[0]: l.split()[1] == "1" for l in r.stdout.splitlines() if l.strip()}

    def _rsync(self, src: str, dst: str) -> float:
        """resumable rsync (--partial); a dropped connection (rc 255/12/30) is retried and resumes."""
        t = time.time()
        for attempt in range(self.RETRIES):
            r = subprocess.run(["rsync", "-rt", "--partial", "--timeout=120", "-e", " ".join(shlex.quote(x) for x in self.ssh[:-1]), src, dst])
            if r.returncode == 0:
                return time.time() - t
            if r.returncode not in (255, 12, 30, 20):
                break
            time.sleep(self.BACKOFF_S * (attempt + 1))
        raise subprocess.CalledProcessError(r.returncode, r.args)

    def push(self, store, asset_ids: list[str]) -> dict[str, float]:
        """Copy the listed assets to the server unless already there. Returns seconds per asset pushed."""
        have = self.has(asset_ids)
        out = {}
        for aid in asset_ids:
            if have.get(aid):
                continue
            a = store.get(aid)
            self.run(f"mkdir -p {shlex.quote(self.root)}/{a.type}/.incoming && rm -rf {shlex.quote(self.root)}/{a.type}/{aid}")
            tmp = f"{self.root}/{a.type}/.incoming/{aid}"
            out[aid] = self._rsync(a.path + "/", f"{self.host}:{tmp}/")
            self.run(f"mv {shlex.quote(tmp)} {shlex.quote(self.root)}/{a.type}/{aid}")   # atomic: appears only when complete
        return out

    def pull(self, store, asset_ids: list[str]) -> dict[str, float]:
        """Copy the listed assets from the server into the local store unless already there."""
        out = {}
        for aid in asset_ids:
            if store.exists(aid):
                continue
            type_ = aid.rsplit("-", 1)[0]
            stage = store.stage_dir(type_)
            out[aid] = self._rsync(f"{self.host}:{self.root}/{type_}/{aid}/", stage + "/")
            final = store.path_of(aid)
            os.makedirs(os.path.dirname(final), exist_ok=True)
            os.replace(stage, final)
        return out
