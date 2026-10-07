"""Content-addressed asset store.

An *asset* is an immutable directory ``<root>/<type>/<id>/`` holding data files plus an
``asset.yaml`` with its provenance.  Two kinds:

* **uploaded**: the id is the sha256 of the file contents (so the same file uploaded twice
  is one asset);
* **computed**: the id is the sha256 of the *recipe* (skill name + version, the ids of its
  input assets, its parameters).  Asking the store for a recipe that already exists returns
  the existing asset instead of recomputing - that is how reuse works across sessions,
  scans and machines (the same recipe has the same id everywhere).

Ids are ``<type>-<12 hex>``; the full hash is kept in ``asset.yaml``.  Writes go to a
temporary directory and are renamed into place, so a crashed job never leaves a half asset.
Nothing in here knows about CT.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable

import yaml

__all__ = ["Asset", "AssetStore", "recipe_hash", "file_hash"]

_CHUNK = 1 << 24


def file_hash(paths: Iterable[str]) -> str:
    """sha256 over the contents of ``paths`` (sorted by basename, so order does not matter)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=os.path.basename):
        h.update(os.path.basename(p).encode()); h.update(b"\0")
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(_CHUNK), b""):
                h.update(block)
    return h.hexdigest()


def recipe_hash(skill: str, version: str, inputs: dict[str, str], params: dict[str, Any]) -> str:
    """sha256 of the canonical recipe (keys sorted, floats as repr, no whitespace)."""
    blob = json.dumps({"skill": skill, "version": version, "inputs": inputs, "params": params},
                      sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Asset:
    id: str
    type: str
    path: str                      # directory
    meta: dict = field(default_factory=dict)

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    @property
    def files(self) -> list[str]:
        return sorted(f for f in os.listdir(self.path) if f != "asset.yaml")

    @property
    def size_bytes(self) -> int:
        return sum(os.path.getsize(self.file(f)) for f in self.files)


class AssetStore:
    """``AssetStore(root)``; the same layout is used on every machine."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    # ----------------------------------------------------------------- lookup
    @staticmethod
    def make_id(type_: str, full_hash: str) -> str:
        return f"{type_}-{full_hash[:12]}"

    def path_of(self, asset_id: str) -> str:
        type_ = asset_id.rsplit("-", 1)[0]
        return os.path.join(self.root, type_, asset_id)

    def exists(self, asset_id: str) -> bool:
        return os.path.isfile(os.path.join(self.path_of(asset_id), "asset.yaml"))

    def get(self, asset_id: str) -> Asset:
        p = self.path_of(asset_id)
        if not self.exists(asset_id):
            raise FileNotFoundError(f"asset {asset_id} not in {self.root}")
        with open(os.path.join(p, "asset.yaml")) as f:
            meta = yaml.safe_load(f)
        return Asset(asset_id, meta["type"], p, meta)

    def find(self, type_: str | None = None) -> list[Asset]:
        out = []
        for t in sorted(os.listdir(self.root)):
            if type_ and t != type_:
                continue
            tdir = os.path.join(self.root, t)
            if not os.path.isdir(tdir):
                continue
            for a in sorted(os.listdir(tdir)):
                if self.exists(a):
                    out.append(self.get(a))
        return out

    def lookup_recipe(self, type_: str, skill: str, version: str, inputs: dict[str, str],
                      params: dict[str, Any]) -> Asset | None:
        """The asset this recipe would produce, if it already exists."""
        aid = self.make_id(type_, recipe_hash(skill, version, inputs, params))
        return self.get(aid) if self.exists(aid) else None

    # ------------------------------------------------------------------ write
    def _commit(self, type_: str, full_hash: str, stage: str, meta: dict) -> Asset:
        aid = self.make_id(type_, full_hash)
        final = self.path_of(aid)
        meta = dict(meta, id=aid, type=type_, hash=full_hash,
                    created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    size_bytes=sum(os.path.getsize(os.path.join(stage, f)) for f in os.listdir(stage)))
        with open(os.path.join(stage, "asset.yaml"), "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        os.makedirs(os.path.dirname(final), exist_ok=True)
        if os.path.exists(final):          # identical recipe/content raced us: keep the existing one
            shutil.rmtree(stage)
        else:
            os.replace(stage, final)
        return self.get(aid)

    def stage_dir(self, type_: str) -> str:
        """A temp directory under the store to write a new asset's files into."""
        os.makedirs(os.path.join(self.root, type_), exist_ok=True)
        return tempfile.mkdtemp(prefix=".stage-", dir=os.path.join(self.root, type_))

    def put_upload(self, type_: str, files: list[str], meta: dict | None = None, move: bool = False) -> Asset:
        """Register user-provided files as an asset (id = hash of their contents)."""
        full = file_hash(files)
        aid = self.make_id(type_, full)
        if self.exists(aid):
            return self.get(aid)
        stage = self.stage_dir(type_)
        for p in files:
            (shutil.move if move else shutil.copy2)(p, os.path.join(stage, os.path.basename(p)))
        return self._commit(type_, full, stage, dict(meta or {}, kind="upload", source_files=[os.path.abspath(p) for p in files]))

    def put_computed(self, type_: str, stage: str, skill: str, version: str, inputs: dict[str, str],
                     params: dict[str, Any], meta: dict | None = None) -> Asset:
        """Commit the files a skill wrote into ``stage`` (from :meth:`stage_dir`) as a computed asset."""
        full = recipe_hash(skill, version, inputs, params)
        return self._commit(type_, full, stage, dict(meta or {}, kind="computed", skill=skill,
                                                       skill_version=version, inputs=inputs, params=params))

    def remove(self, asset_id: str) -> None:
        shutil.rmtree(self.path_of(asset_id))
