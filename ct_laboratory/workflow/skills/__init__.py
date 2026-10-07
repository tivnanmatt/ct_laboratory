"""Skills: thin wrappers that turn ct_laboratory operations into recorded jobs.

A skill is ``fn(config, session, job) -> {"outputs": {role: asset_id}, "metrics": {...}}``,
registered under a dotted name.  The YAML config is the only interface - the ``ctlab`` CLI,
an AI agent or a person all write the same YAML - and the physics and numerics live in the
rest of ct_laboratory.  Scanner projects (e.g. research-ring for StaticCT) build the input
assets with their own step code and submit jobs here, locally or on a remote server.

Asset types / session roles:
  geometry        StepAndShootGeometry (geometry.pt)
  sinogram        sinogram.pt: y [n_rot, n_ray] line integrals, w [n_rot, n_ray] weights/mask, n_view, n_u, n_v
  eigen@<nx>      window eigenbasis for grid nx (eigen.pt; k in asset.yaml)
  recon@<nx>      volume_<nx>.pt per cascade level + metrics.json
"""
from __future__ import annotations

from typing import Callable

SKILLS: dict[str, Callable] = {}
SKILL_VERSION = "1"


def skill(name: str):
    def deco(fn):
        SKILLS[name] = fn; fn.skill_name = name
        return fn
    return deco


def code_params(extra_repos: dict[str, str] | None = None) -> dict:
    """Code fingerprint that goes into every computed recipe, so an asset is tied to the exact
    code that produced it (ct_laboratory, plus any repos registered via register_code_repo)."""
    from ..session import code_fingerprints
    return {k: v["fingerprint"] for k, v in code_fingerprints({**CODE_REPOS, **(extra_repos or {})}).items()}


CODE_REPOS: dict[str, str] = {}


def register_code_repo(name: str, path: str) -> None:
    """Client projects register their checkout so their commit enters recipe hashes too."""
    CODE_REPOS[name] = path


from . import assets_io, correction, eigen, projector, recon  # noqa: E402,F401  (registers the skills)
