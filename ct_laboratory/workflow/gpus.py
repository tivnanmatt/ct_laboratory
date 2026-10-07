"""GPU discovery: every GPU step asks :func:`available_devices` and uses all of them."""
from __future__ import annotations

import os
import subprocess

import torch

__all__ = ["available_devices", "gpu_info"]


def available_devices(max_gpus: int | None = None) -> list[str]:
    """``['cuda:0', 'cuda:1', ...]`` for the visible GPUs (honours ``CUDA_VISIBLE_DEVICES``),
    capped at ``max_gpus`` or the ``CT_MAX_GPUS`` env var; ``['cpu']`` when there is none."""
    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    cap = max_gpus if max_gpus is not None else int(os.environ.get("CT_MAX_GPUS", "0") or 0)
    if cap > 0:
        n = min(n, cap)
    return [f"cuda:{i}" for i in range(n)]


def gpu_info() -> list[str]:
    """``nvidia-smi`` one-liners (name, driver, memory) or ``[]``."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return [l.strip() for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []
    except Exception:
        return []
