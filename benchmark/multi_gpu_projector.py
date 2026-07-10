#!/usr/bin/env python3
"""Ray-sharded multi-GPU wrapper around CTProjector3DModule — no kernel changes.

Rays are independent, so each GPU gets a contiguous ray shard and its own
sub-projector (uint16-precomputed). The volume (~67MB at res=512) is small
enough to replicate per device. Forward gathers sinogram shards; backprojection
sums per-device volumes. Each device is driven from its own Python thread so
any internal synchronization in the module can't serialize the GPUs.

Run as a script for a 1-GPU vs 2-GPU scaling test:
    python multi_gpu_projector.py --resolution 512 --repeats 5
"""
import argparse
import threading
import time

import torch

from ct_laboratory.ct_projector_3d_module import CTProjector3DModule, precompute_tvals_stitched

CASE_SPECS = {
    64: {"nz": 8, "spacing_mm": 6.4},
    128: {"nz": 16, "spacing_mm": 3.2},
    256: {"nz": 32, "spacing_mm": 1.6},
    512: {"nz": 64, "spacing_mm": 0.8},
}
B_SHIFT_Z = 5.0
PRECOMPUTE_CHUNK = 32768


def volume_grid(res, device):
    spec = CASE_SPECS[res]
    M = torch.eye(3, device=device) * spec["spacing_mm"]
    b = torch.tensor([
        -(res - 1) / 2.0 * spec["spacing_mm"],
        -(res - 1) / 2.0 * spec["spacing_mm"],
        -(spec["nz"] - 1) / 2.0 * spec["spacing_mm"] + B_SHIFT_Z,
    ], device=device)
    return M, b, spec["nz"]


def _run_threads(fns):
    """Run one callable per device concurrently; re-raise the first error."""
    errs = []
    def wrap(fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ts = [threading.Thread(target=wrap, args=(fn,)) for fn in fns]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]


class MultiGPUProjector3D:
    """Shards rays across devices; one uint16-precomputed sub-projector each."""

    def __init__(self, res, src, dst, devices, verbose=True):
        self.res = res
        self.devices = [torch.device(d) for d in devices]
        self._pin_cache = {}
        n_ray = src.shape[0]
        shard = (n_ray + len(self.devices) - 1) // len(self.devices)
        self.bounds = [(i * shard, min((i + 1) * shard, n_ray)) for i in range(len(self.devices))]
        self.subs = [None] * len(self.devices)

        def build(i):
            dev = self.devices[i]
            # The CUDA extension launches kernels on the thread's current
            # device; without this, cuda:1 work launches in cuda:0's context
            # and faults with illegal memory access.
            torch.cuda.set_device(dev)
            lo, hi = self.bounds[i]
            M, b, nz = volume_grid(res, dev)
            s, d = src[lo:hi].contiguous(), dst[lo:hi].contiguous()
            tvals = precompute_tvals_stitched(
                res, res, nz, M.cpu(), b.cpu(), s, d,
                chunk_size=PRECOMPUTE_CHUNK, backend="cuda", device=dev,
                verbose=False, use_compression=True)
            self.subs[i] = CTProjector3DModule(
                res, res, nz, M, b, s.to(dev), d.to(dev),
                backend="cuda", device=dev, precomputed_intersections=False,
                tvals=tvals, use_compression=True).to(dev)
            if verbose:
                print(f"  shard {i}: rays [{lo}:{hi}] on {dev}", flush=True)

        _run_threads([lambda i=i: build(i) for i in range(len(self.devices))])

    # Cross-device transfers are staged through cached pinned host buffers.
    # Direct gpu->gpu .to() uses PCIe peer-to-peer when the driver advertises
    # it, and on some multi-GPU hosts (observed on a cloud 8x L40S box) P2P is
    # advertised but broken: copies return corrupted data and run slowly.
    # Device 0's own shard skips staging entirely (its data never crosses
    # devices), so the 1-GPU path has zero overhead.
    def _pinned(self, key, like):
        buf = self._pin_cache.get(key)
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            buf = torch.empty(like.shape, dtype=like.dtype, device="cpu", pin_memory=True)
            self._pin_cache[key] = buf
        buf.copy_(like)
        return buf

    def forward_project(self, volume):
        """volume on devices[0] -> sinogram on devices[0]."""
        outs = [None] * len(self.devices)
        vol_cpu = self._pinned("vol", volume) if len(self.devices) > 1 else None

        def fwd(i):
            torch.cuda.set_device(self.devices[i])
            vol_i = volume if i == 0 else vol_cpu.to(self.devices[i], non_blocking=True)
            outs[i] = self.subs[i].forward_project(vol_i)
            torch.cuda.synchronize(self.devices[i])

        _run_threads([lambda i=i: fwd(i) for i in range(len(self.devices))])
        if len(self.devices) == 1:
            return outs[0]
        return torch.cat([outs[0]] + [o.cpu().to(self.devices[0]) for o in outs[1:]])

    def back_project(self, sino):
        """sino on devices[0] -> volume on devices[0] (summed across shards)."""
        vols = [None] * len(self.devices)
        sino_cpu = self._pinned("sino", sino) if len(self.devices) > 1 else None

        def bak(i):
            torch.cuda.set_device(self.devices[i])
            lo, hi = self.bounds[i]
            sino_i = sino[lo:hi] if i == 0 else sino_cpu[lo:hi].to(self.devices[i], non_blocking=True)
            v = self.subs[i].back_project(sino_i)
            if i > 0:
                # pinned D2H inside the worker so shard copies run in parallel
                v = self._pinned(f"volout{i}", v)
            vols[i] = v
            torch.cuda.synchronize(self.devices[i])

        _run_threads([lambda i=i: bak(i) for i in range(len(self.devices))])
        acc = vols[0]
        for v in vols[1:]:
            acc = acc + v.to(self.devices[0], non_blocking=True)
        torch.cuda.synchronize(self.devices[0])
        return acc


def sync_all(devices):
    for d in devices:
        torch.cuda.synchronize(d)


def median_ms(fn, devices, repeats):
    vals = []
    out = None
    for _ in range(repeats):
        sync_all(devices)
        t0 = time.perf_counter()
        out = fn()
        sync_all(devices)
        vals.append((time.perf_counter() - t0) * 1000)
    vals.sort()
    return vals[len(vals) // 2], out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resolution", type=int, default=512, choices=sorted(CASE_SPECS))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--geometry", default="/opt/benchmark/geometry/geometry_80src_3rot_240view.pt")
    args = p.parse_args()
    res = args.resolution

    blob = torch.load(args.geometry, map_location="cpu", weights_only=True)
    src, dst = blob["src"], blob["dst"]
    n_gpu = torch.cuda.device_count()
    print(f"res={res}, rays={src.shape[0]}, GPUs={n_gpu}: "
          + ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu)), flush=True)

    counts = [c for c in (1, 2, 4, 8) if c <= n_gpu]
    nz = CASE_SPECS[res]["nz"]
    vol = torch.randn(res, res, nz, device="cuda:0")
    ref = {}
    results = []
    for k in counts:
        devs = [f"cuda:{i}" for i in range(k)]
        print(f"[{k}-GPU] building sharded projector...", flush=True)
        t0 = time.perf_counter()
        proj = MultiGPUProjector3D(res, src, dst, devs)
        build_s = time.perf_counter() - t0
        print(f"  built in {build_s:.1f}s", flush=True)
        fwd, sino = median_ms(lambda: proj.forward_project(vol), devs, args.repeats)
        back, volb = median_ms(lambda: proj.back_project(sino), devs, args.repeats)
        if k == 1:
            ref = {"fwd": fwd, "back": back, "sino": sino, "vol": volb}
            print(f"  {k}-GPU: fwd={fwd:.1f}ms back={back:.1f}ms", flush=True)
        else:
            se = (ref["sino"] - sino).abs().max().item() / max(ref["sino"].abs().max().item(), 1e-9)
            ve = (ref["vol"] - volb).abs().max().item() / max(ref["vol"].abs().max().item(), 1e-9)
            print(f"  {k}-GPU: fwd={fwd:.1f}ms back={back:.1f}ms | "
                  f"scaling fwd {ref['fwd']/fwd:.2f}x back {ref['back']/back:.2f}x "
                  f"(eff {ref['fwd']/fwd/k*100:.0f}%/{ref['back']/back/k*100:.0f}%) | "
                  f"maxrelerr sino={se:.1e} vol={ve:.1e}", flush=True)
        results.append((k, fwd, back, build_s))
        del proj
        torch.cuda.empty_cache()

    print("\nGPUs  fwd_ms  back_ms  build_s", flush=True)
    for k, fwd, back, build_s in results:
        print(f"{k:>4} {fwd:>7.1f} {back:>8.1f} {build_s:>8.1f}", flush=True)


if __name__ == "__main__":
    main()
