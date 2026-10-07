"""Workflow area - generic bookkeeping for reconstruction studies (no CT physics here):

  assets    content-addressed, immutable asset store (uploads hashed by content, computed
            assets hashed by recipe -> automatic reuse)
  session   sessions (role -> asset id) and recorded jobs (config, status, log, metrics)
  gpus      GPU discovery; every GPU step runs on all visible GPUs
  remote    ssh/rsync transport: push the assets a server lacks, pull results back
  skills    recorded-job wrappers (asset import/export, eigen.compute/sweep, recon.cascade)
  cli       the ctlab command (local or --remote execution); server_bootstrap.sh turns a
            fresh GPU pod into a server (health-gated)
"""
from .assets import Asset, AssetStore, file_hash, recipe_hash
from .gpus import available_devices, gpu_info
from .session import JobRecord, Session, code_fingerprints, code_versions, run_job
from .remote import RemoteStore
from . import skills

__all__ = ["Asset", "AssetStore", "file_hash", "recipe_hash", "available_devices", "gpu_info",
           "JobRecord", "Session", "code_fingerprints", "code_versions", "run_job", "RemoteStore", "skills"]
