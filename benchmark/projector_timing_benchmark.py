#!/usr/bin/env python3
"""Projector timing/memory benchmark for the 80-source 3-rotation (240-view) system.

Benchmarks forward/back projection of CTProjector3DModule at one resolution
across backend x precompute configurations, recording wall time and memory.

Geometry is loaded from a plain {src, dst} tensor file (see
extract_geometry.py), so this script only needs ct_laboratory installed --
no experiment-specific pickles.

Results go to --output-dir as CSV + JSON, and the JSON summary is printed to
stdout so `docker logs` is enough to collect results from remote deployments.
"""
import argparse
import csv
import json
import os
import platform
import resource
import socket
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ct_laboratory.ct_projector_3d_module import CTProjector3DModule, precompute_tvals_stitched
from ct_laboratory.ct_projector_3d_torch import (
    back_project_3d_torch,
    compute_intersections_3d_torch,
    forward_project_3d_torch,
)

# CTProjector3DModule(precomputed_intersections=True) computes intersections
# for all rays in one unchunked CPU call, which OOMs host RAM at higher
# resolutions (e.g. res=250, 2.95M rays). precompute_tvals_stitched does the
# same computation in memory-safe chunks -- always precompute through it
# instead of letting the constructor do it internally.
PRECOMPUTE_CHUNK_SIZE = 32768

# Volume cases matching step0_projectors.py in 20260623_sparse_eig_3d
# (400 mm XY FOV, 60 mm Z FOV, +5 mm z shift).
CASE_SPECS = {
    64: {"nz": 8, "spacing_mm": 6.4},
    128: {"nz": 16, "spacing_mm": 3.2},
    250: {"nz": 20, "spacing_mm": 2.0},
    256: {"nz": 32, "spacing_mm": 1.6},
    512: {"nz": 64, "spacing_mm": 0.8},
    1024: {"nz": 128, "spacing_mm": 0.4},
}
B_SHIFT_Z = 5.0

# "backend" here means cuda kernels vs pure-PyTorch on GPU vs pure-PyTorch on
# CPU (cuda and torch-gpu share the physical GPU). Each maps to a list of
# (label, module_backend, precompute_mode) configs.
BACKEND_CONFIGS = {
    "cuda": [
        ("CUDA no precompute", "cuda", "none"),
        ("CUDA precomputed uint16", "cuda", "compressed"),
        ("CUDA precomputed float", "cuda", "full"),
    ],
    "torch-gpu": [
        ("Torch GPU no precompute", "torch", "none"),
        ("Torch GPU precomputed uint16", "torch", "compressed"),
        ("Torch GPU precomputed float", "torch", "full"),
    ],
    "torch-cpu": [
        ("Torch CPU no precompute", "torch", "none"),
        ("Torch CPU precomputed float", "torch", "full"),
    ],
}


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def gpu_peak_mb(device):
    if torch.device(device).type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def median_time_ms(fn, device, repeats):
    vals = []
    out = None
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        vals.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(vals)), out


def load_geometry(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return blob["src"].to(device), blob["dst"].to(device)


def volume_grid(res, device):
    spec = CASE_SPECS[res]
    M = torch.eye(3, device=device) * spec["spacing_mm"]
    b = torch.tensor(
        [
            -(res - 1) / 2.0 * spec["spacing_mm"],
            -(res - 1) / 2.0 * spec["spacing_mm"],
            -(spec["nz"] - 1) / 2.0 * spec["spacing_mm"] + B_SHIFT_Z,
        ],
        device=device,
    )
    return M, b, spec["nz"]


def projector_bytes(proj):
    total = 0
    for name in ("M", "b", "src", "dst", "tvals_uint16", "tvals_start", "tvals_scale", "tvals_full"):
        t = getattr(proj, name, None)
        if isinstance(t, torch.Tensor):
            total += t.numel() * t.element_size()
    return total


def build_projector(res, backend, device, precompute_mode, src, dst):
    M, b, nz = volume_grid(res, device)
    use_compression = precompute_mode == "compressed"
    t0 = time.perf_counter()
    if precompute_mode == "none":
        proj = CTProjector3DModule(
            res, res, nz, M, b, src, dst,
            backend=backend, device=torch.device(device),
            precomputed_intersections=False,
        ).to(device)
    else:
        # Chunked precompute (memory-safe at high resolution) instead of
        # letting the constructor compute all-rays-at-once on CPU.
        tvals = precompute_tvals_stitched(
            res, res, nz, M.cpu(), b.cpu(), src.cpu(), dst.cpu(),
            chunk_size=PRECOMPUTE_CHUNK_SIZE,
            backend=backend if backend == "cuda" else "torch",
            device=torch.device(device),
            verbose=False,
            use_compression=use_compression,
        )
        proj = CTProjector3DModule(
            res, res, nz, M, b, src, dst,
            backend=backend, device=torch.device(device),
            precomputed_intersections=False,
            tvals=tvals,
            use_compression=use_compression,
        ).to(device)
    sync(device)
    build_ms = (time.perf_counter() - t0) * 1000.0
    return proj, build_ms


# compute_intersections_3d_torch materializes several [n_ray, n_planes]
# tensors at once (plus torch.sort's int64 indices = 2x the float32 size), so
# calling it on all rays OOMs at high resolution. Rays are independent, so the
# on-the-fly path chunks over rays: same math, bounded memory. Timing wraps
# the whole chunked loop, so measured cost still includes intersection compute.
ONTHEFLY_CHUNK_SIZE = 131072


def forward_project_on_the_fly(vol, M, b, src, dst, res, nz):
    n_ray = src.shape[0]
    out = torch.empty(n_ray, device=vol.device, dtype=vol.dtype)
    for i in range(0, n_ray, ONTHEFLY_CHUNK_SIZE):
        s, d = src[i:i + ONTHEFLY_CHUNK_SIZE], dst[i:i + ONTHEFLY_CHUNK_SIZE]
        tv = compute_intersections_3d_torch(res, res, nz, M, b, s, d)
        out[i:i + ONTHEFLY_CHUNK_SIZE] = forward_project_3d_torch(vol, tv, M, b, s, d)
        del tv
    return out


def back_project_on_the_fly(sino, M, b, src, dst, res, nz):
    n_ray = src.shape[0]
    vol = torch.zeros(res, res, nz, device=sino.device, dtype=sino.dtype)
    for i in range(0, n_ray, ONTHEFLY_CHUNK_SIZE):
        s, d = src[i:i + ONTHEFLY_CHUNK_SIZE], dst[i:i + ONTHEFLY_CHUNK_SIZE]
        tv = compute_intersections_3d_torch(res, res, nz, M, b, s, d)
        vol += back_project_3d_torch(sino[i:i + ONTHEFLY_CHUNK_SIZE], tv, M, b, s, d, res, res, nz)
        del tv
    return vol


def run_config(label, backend, mode, res, device, repeats, src_cpu, dst_cpu):
    row = dict(configuration=label, backend=backend, device=str(device),
               precomputed=mode, resolution=res, nz=CASE_SPECS[res]["nz"],
               build_ms=np.nan, forward_ms=np.nan, backproject_ms=np.nan,
               peak_gpu_mb=np.nan, rss_mb=np.nan, projector_memory_mb=np.nan,
               n_rays=np.nan, status="ok", traceback="")
    try:
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.set_device(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        src, dst = src_cpu.to(dev), dst_cpu.to(dev)
        nz = CASE_SPECS[res]["nz"]
        vol = torch.randn(res, res, nz, device=dev)
        if dev.type == "cpu":
            repeats = 1  # keep CPU sweeps tractable at high resolution
        if backend == "torch" and mode == "none":
            M, b, _ = volume_grid(res, dev)
            fwd_ms, sino = median_time_ms(
                lambda: forward_project_on_the_fly(vol, M, b, src, dst, res, nz), dev, 1)
            back_ms, _ = median_time_ms(
                lambda: back_project_on_the_fly(sino, M, b, src, dst, res, nz), dev, 1)
            row.update(
                build_ms=0.0, forward_ms=fwd_ms, backproject_ms=back_ms,
                peak_gpu_mb=gpu_peak_mb(dev), rss_mb=rss_mb(),
                projector_memory_mb=(src.numel() * src.element_size()
                                     + dst.numel() * dst.element_size()) / 1e6,
                n_rays=int(sino.numel()),
            )
        else:
            proj, build_ms = build_projector(res, backend, dev, mode, src, dst)
            row["build_ms"] = build_ms
            fwd_ms, sino = median_time_ms(lambda: proj.forward_project(vol), dev, repeats)
            back_ms, _ = median_time_ms(lambda: proj.back_project(sino), dev, repeats)
            row.update(
                forward_ms=fwd_ms, backproject_ms=back_ms,
                peak_gpu_mb=gpu_peak_mb(dev), rss_mb=rss_mb(),
                projector_memory_mb=projector_bytes(proj) / 1e6,
                n_rays=int(sino.numel()),
            )
            del proj
        del src, dst, vol
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        row["status"] = f"error: {type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()
    return row


def host_info(device):
    info = dict(
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        dev = torch.device(device)
        idx = dev.index or 0
        info.update(
            cuda_version=torch.version.cuda,
            gpu_name=torch.cuda.get_device_name(idx),
            gpu_count=torch.cuda.device_count(),
            gpu_total_mem_gb=torch.cuda.get_device_properties(idx).total_memory / 1e9,
        )
    return info


def write_outputs(outdir, meta, rows):
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "rows": rows}
    json_path = outdir / "projector_timing.json"
    json_path.write_text(json.dumps(payload, indent=2))
    csv_path = outdir / "projector_timing.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # Keep shared-drive/bind-mount outputs writable by the host user
    # regardless of what uid the container ran as.
    for p in (outdir, json_path, csv_path):
        try:
            os.chmod(p, 0o777 if p.is_dir() else 0o666)
        except OSError:
            pass
    return payload


def main():
    global PRECOMPUTE_CHUNK_SIZE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolutions", type=int, nargs="+", default=[64, 128, 256, 512],
                   choices=sorted(CASE_SPECS),
                   help="in-plane voxel counts to sweep (nz/spacing implied)")
    p.add_argument("--backends", nargs="+", default=["cuda"],
                   choices=sorted(BACKEND_CONFIGS),
                   help="cuda = custom kernels, torch-gpu = pure PyTorch on GPU, torch-cpu = pure PyTorch on CPU")
    p.add_argument("--torch-max-res", type=int, default=64,
                   help="torch-gpu/torch-cpu backends only run at resolutions <= this "
                        "(they are too slow above; extrapolate from the low-res point)")
    p.add_argument("--precompute-modes", nargs="+", default=["none", "compressed", "full"],
                   choices=["none", "compressed", "full"],
                   help="only run configs with these precompute modes (compressed = uint16)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--geometry", default=str(Path(__file__).parent / "geometry" / "geometry_80src_3rot_240view.pt"))
    p.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    p.add_argument("--tag", default="", help="free-form label recorded in output (e.g. node name)")
    p.add_argument("--precompute-chunk-size", type=int, default=PRECOMPUTE_CHUNK_SIZE,
                   help="rays per chunk when precomputing tvals; lower this if a node OOMs")
    args = p.parse_args()
    PRECOMPUTE_CHUNK_SIZE = args.precompute_chunk_size

    blob = torch.load(args.geometry, map_location="cpu", weights_only=True)
    src_cpu, dst_cpu = blob["src"], blob["dst"]

    meta = dict(
        resolutions=args.resolutions,
        backends=args.backends,
        torch_max_res=args.torch_max_res,
        n_rays=int(src_cpu.shape[0]),
        repeats=args.repeats,
        tag=args.tag,
        geometry=args.geometry,
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
        **host_info(args.device),
    )
    print(json.dumps({"meta": meta}, indent=2), flush=True)

    outdir = Path(args.output_dir) / f"sweep_{args.tag or meta['hostname']}"
    plan = []
    for res in args.resolutions:
        for bk in args.backends:
            if bk != "cuda" and res > args.torch_max_res:
                continue  # torch backends too slow above torch_max_res
            for label, module_backend, mode in BACKEND_CONFIGS[bk]:
                if mode not in args.precompute_modes:
                    continue
                device = "cpu" if bk == "torch-cpu" else args.device
                plan.append((label, module_backend, mode, res, device))

    rows = []
    for label, module_backend, mode, res, device in plan:
        print(f"[bench] {label} (res={res}, device={device}) ...", flush=True)
        row = run_config(label, module_backend, mode, res, device, args.repeats, src_cpu, dst_cpu)
        status = row["status"] if row["status"] != "ok" else (
            f"fwd={row['forward_ms']:.2f}ms back={row['backproject_ms']:.2f}ms "
            f"build={row['build_ms']:.0f}ms peak_gpu={row['peak_gpu_mb']:.0f}MB")
        print(f"[bench]   -> {status}", flush=True)
        rows.append(row)
        write_outputs(outdir, meta, rows)  # incremental, survives OOM kills

    payload = write_outputs(outdir, meta, rows)
    print("RESULTS_JSON_BEGIN", flush=True)
    print(json.dumps(payload), flush=True)
    print("RESULTS_JSON_END", flush=True)
    print(f"wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
