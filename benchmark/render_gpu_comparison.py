#!/usr/bin/env python3
"""Render the cross-GPU forward/backprojection comparison chart from
results_archive/*.json (res512, CUDA backend, uint16-compressed tvals).

Writes benchmark/figures/gpu_comparison_res512.png.

Example:
  /opt/venv/bin/python benchmark/render_gpu_comparison.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "results_archive"
OUT = HERE / "figures" / "gpu_comparison_res512.png"


def headline_row(blob):
    """res512, cuda backend, compressed uint16 precompute row (or None)."""
    for row in blob.get("rows", []):
        if (row.get("resolution") == 512 and row.get("backend") == "cuda"
                and row.get("precomputed") in ("compressed", True, "uint16")
                and row.get("status", "ok") == "ok"
                and row.get("forward_ms") is not None):
            return row
    return None


def main():
    entries = []
    for p in sorted(ARCHIVE.glob("*.json")):
        blob = json.loads(p.read_text())
        row = headline_row(blob)
        if row is None:
            continue
        gpu = blob.get("meta", {}).get("gpu_name", p.stem)
        entries.append((p.stem, gpu, float(row["forward_ms"]),
                        float(row["backproject_ms"])))
    # de-duplicate identical GPU names, keep the fastest forward
    best = {}
    for stem, gpu, fwd, bak in entries:
        key = f"{gpu} ({stem})" if gpu in {g for _, g, _, _ in entries
                                           if sum(1 for e in entries if e[1] == gpu) > 1} else gpu
        best[key] = (fwd, bak)
    items = sorted(best.items(), key=lambda kv: kv[1][0])

    labels = [k for k, _ in items]
    fwd = np.array([v[0] for _, v in items])
    bak = np.array([v[1] for _, v in items])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(labels) + 2.2))
    h = 0.38
    ax.barh(y - h / 2, fwd, h, label="Forward projection", color="#1976D2")
    ax.barh(y + h / 2, bak, h, label="Backprojection", color="#FFA726")
    for yi, (f, b) in enumerate(zip(fwd, bak)):
        ax.text(f + 3, yi - h / 2, f"{f:.0f}", va="center", fontsize=8)
        ax.text(b + 3, yi + h / 2, f"{b:.0f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Median wall time per application (ms)")
    ax.set_title("StaticCT projector — res512 (512×512×64), 2,949,120 rays\n"
                 "CUDA backend, uint16-compressed precomputed tvals",
                 fontsize=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.25)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}  ({len(labels)} GPUs)")


if __name__ == "__main__":
    main()
