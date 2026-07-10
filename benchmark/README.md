# StaticCT Projector Timing Benchmark

Multi-GPU timing and memory benchmark for the `ct_laboratory` 3D CT projector
(`CTProjector3DModule`) on the full StaticCT ring geometry. This directory
contains the benchmark driver, a RunPod cloud sweep orchestrator, a
multi-GPU ray-sharded projector wrapper, and an archive of results across
16 GPU configurations.

## Motivation

Iterative CT reconstruction is dominated by repeated application of the
forward projector `A` and backprojector `Aᵀ`. Every MBIR/PCG iteration costs
at least one forward + one backprojection, so the per-application wall time
of these two operators — at clinically relevant volume sizes — directly sets
the reconstruction latency budget and determines which GPUs can serve the
system in production. We benchmarked many GPUs (consumer, workstation, and
datacenter) and several projector configurations to answer three questions:

1. **How fast is one projector application** at each volume resolution, per
   GPU model — the number that multiplies into every iterative algorithm?
2. **What does precomputation buy?** The projector can precompute ray/voxel
   intersection parameters ("tvals") once and reuse them every call. We
   compare no-precompute (recompute intersections on the fly), compressed
   uint16 precompute, and full float32 precompute — a speed/memory trade.
3. **Which hardware should we buy or rent?** The same containerized
   benchmark ran locally (AXIS03, DGX A100 nodes) and on rented RunPod
   instances (RTX 4090/5090, L40S, A100 80GB, H100 variants, ...), making
   the numbers directly comparable across the fleet.

## Methodology

**Geometry.** All runs use the real calibrated StaticCT ring geometry,
extracted once by [`extract_geometry.py`](extract_geometry.py) from the
pickled `StaticCTProjector3D` instance into a plain-tensor file
[`geometry/geometry_80src_3rot_240view.pt`](geometry/) (loadable with
`torch.load(weights_only=True)`, no pickle dependency):

- 80 sources × 3 gantry rotations = **240 views**
- 12,288 detector pixels per view (16 active modules × 48 columns × 16 rows)
- **2,949,120 rays total** — identical for every volume resolution

**Volume grids.** Fixed 400 mm transaxial FOV and 60 mm axial extent
(+5 mm z shift), so resolution changes voxel pitch, not coverage:

| Resolution | n_z | Voxel pitch (mm) | Voxels |
|-----------:|----:|-----------------:|-------:|
| 64  | 8   | 6.4 | 32,768 |
| 128 | 16  | 3.2 | 262,144 |
| 256 | 32  | 1.6 | 2,097,152 |
| 512 | 64  | 0.8 | 16,777,216 |

(The driver also defines 250/1024 grids for special cases.)

**Timing.** [`projector_timing_benchmark.py`](projector_timing_benchmark.py)
builds the projector for each (resolution × backend × precompute mode)
configuration, then times `forward_project` and `back_project` separately:
CUDA-synchronize before and after each call, record wall ms, and report the
**median over the repeats** (default 5 locally, 3 in the cloud sweep; the
median absorbs first-call warmup). One-time projector build /
tvals-precompute time (`build_ms`), peak GPU memory, resident projector
memory, and host RSS are recorded alongside. Results are written as CSV +
JSON and also printed to stdout so `docker logs` capture them.

**Configurations.** Backends: custom `cuda` kernels (primary), `torch-gpu`,
`torch-cpu` (small grids only). Precompute modes: `none` (intersections
recomputed every call, chunked), `compressed` (uint16-quantized tvals), and
`full` (float32 tvals).

**Cloud sweep.** [`runpod_sweep.py`](runpod_sweep.py) fans the benchmark
across RunPod GPU types in parallel: create pod → wait for SSH → run the
containerized benchmark (`--resolutions 64 128 256 512 --backends cuda
--precompute-modes compressed --repeats 3`) → scp the JSON back → always
delete the pod. [`Dockerfile`](Dockerfile) / [`docker-compose.yml`](docker-compose.yml)
pin the environment.

**Multi-GPU scaling.** [`multi_gpu_projector.py`](multi_gpu_projector.py)
shards the 2.9M rays contiguously across GPUs, each shard with its own
uint16-precomputed sub-projector and its own driver thread; the volume
(~67 MB at res512) is replicated per device. Forward gathers sinogram
shards; backprojection sums per-device partial volumes. This wrapper is
what the reconstruction pipeline uses to hold res512 on 2×24 GB cards.

## Results

Headline configuration: **res512 (512×512×64), CUDA backend, uint16
precomputed tvals** — the configuration the reconstruction pipeline runs in
production. Projector resident memory ≈ 6.5 GB; peak ≈ 6.8 GB. Median of
repeats; one forward / one backprojection of all 2,949,120 rays.

Single-GPU and ray-sharded multi-GPU configurations in one table, sorted by
forward-projection time. Multi-GPU medians are from `multi_gpu_projector.py`
runs at res512 only, recorded in `lab-ops/results/benchmark-2026-07-04.md`
(the scaling script prints to stdout rather than writing archive JSONs).

| GPU configuration | Forward (ms) | Backproject (ms) | Build (s) | Iter pairs/min @512 |
|-------------------|-------------:|-----------------:|----------:|--------------------:|
| 8× L40S (sharded) | **16.5** | 33.7 | ≈12.7 | 1195 |
| 4× L40S (sharded) | 21.4 | **28.0** | ≈20 | **1215** |
| 2× L40S (sharded) | 32.6 | 38.6 | ≈39 | 843 |
| 4× RTX 4090 (sharded, community) | 34.4 | 39.4 | ≈20 | 813 |
| 2× RTX 4090 (sharded, AXIS03) | 39.1 | 47.0 | ≈38 | 697 |
| H100 80GB HBM3 | 46.2 | 36.1 | 99.1 | 729 |
| 2× RTX 4090 (sharded, community) | 48.9 | 46.1 | ≈40 | 632 |
| H100 NVL | 49.3 | 30.9 | 101.9 | 748 |
| RTX 5090 | 49.4 | 37.8 | 69.0 | 688 |
| L40S | 50.3 | 64.0 | 77.7 | 525 |
| H100 PCIe | 51.8 | 42.9 | 105.5 | 634 |
| RTX 4090 (AXIS03, cuda:0) | 52.7 | 71.7 | 74.9 | 482 |
| RTX 4090 (RunPod) | 52.8 | 71.5 | 80.7 | 483 |
| RTX PRO 6000 Blackwell | 60.9 | 33.0 | 85.5 | 639 |
| A100-SXM4-80GB | 72.3 | 70.6 | 125.3 | 420 |
| A100 80GB PCIe | 72.5 | 71.1 | 124.2 | 418 |
| A100-SXM4-40GB (DGX gpu2/7/8) | 72.7–72.9 | 77.4–78.0 | 124–130 | 399 |
| A100-SXM4-40GB (gpu5, loaded) | 101.0 | 109.0 | 170.7 | — |
| A100-SXM4-40GB (gpu3, loaded) | 145.5 | 154.5 | 197.7 | — |
| RTX 3090 | 355.0 | 495.5 | 115.7 | 71 |
| RTX A5000 | 392.0 | 474.4 | 135.0 | 69 |

"Iter pairs/min" = forward+backprojection pairs per minute at res512, the unit
of iterative-reconstruction work. Sharded build times parallelize across
shards (measured 78 s → 12.7 s at 8 GPUs; intermediate counts estimated ∝ 1/N).

Failures worth knowing: one A100-40GB node (gpu1) OOM'd at res256/512 with
uint16 precompute while other jobs held memory (it completed res64:
14.3/10.3 ms and res128: 20.2/13.1 ms); the A40 and RTX 6000 Ada RunPod
images failed on too-old NVIDIA drivers before benchmarking.

**Memory strategy for multi-GPU.** Rays are independent, so the wrapper gives
each GPU a contiguous shard of the 2,949,120 rays with its OWN sub-projector:
the uint16 tvals are row-sliced per shard, so an N-GPU split holds ~6.5/N GB
of projector weights per card. The volume (~67 MB at res512) is small enough
to replicate on every device. Forward projection runs each shard on its own
Python thread and gathers the sinogram pieces on device 0; backprojection
produces one partial volume per device and sums them. Cross-GPU traffic is
host-staged through pinned CPU buffers (robust on hosts with broken PCIe
peer-to-peer), which is also why scaling saturates: forward reaches 3.1× at
8 GPUs, backprojection peaks at 4 GPUs (2.3×) where the per-device volume
gathers start to dominate. Sweet spot: 4 GPUs at res512. Weight-build time
scales near-perfectly (78 s → 12.7 s on 8 GPUs). Value analysis: multi-GPU
buys latency, not value — iterations/min/$ drops with GPU count (L40S
530 → 426 → 307 → 151; 4090 1346 → 929 → 598), so single community 4090s
remain the cheapest iteration supply and sharding is for latency-critical
paths.

**Resolution scaling** (RTX 4090 AXIS03, CUDA + uint16, fwd/back ms):
res64 ≈ single-digit ms up through **res512 = 52.7 / 71.7 ms**. The
no-precompute path at res512 costs **6183 / 4536 ms** — precomputing
intersections is a ~100× speedup at the cost of holding 6.5 GB of tvals.
Full float32 tvals double the memory (13.2 GB peak) and were *slower* on
the 4090 (68.8 / 99.1 ms) — bandwidth-bound, so the compressed uint16
format wins on both axes.

**Interpretation.**

- One res512 projector application costs ~50 ms forward / 31–72 ms back on
  any modern high-bandwidth GPU. A full preconditioned-CG iteration
  (1 fwd + 1 bak) therefore costs ≈ 0.1–0.15 s of projection compute.
- Ranking follows memory bandwidth, not FLOPs: HBM3 H100s and the
  GDDR7 RTX 5090 lead; Ampere (A100, RTX 3090/A5000) trails; the 4090
  matches datacenter cards on forward but loses on backprojection.
- Consumer RTX 4090s are ~1.1–2× the best datacenter parts at ~1/10 the
  cost, which is why the reconstruction pipeline targets 2×4090 with the
  ray-sharded multi-GPU wrapper.
- Build time (~70–130 s) is a one-time cache: the reconstruction pipeline
  saves the uint16 tvals to disk (step 0) and reloads in ~seconds.

## Files

| File | Purpose |
|------|---------|
| `projector_timing_benchmark.py` | Benchmark driver (timing + memory per config) |
| `runpod_sweep.py` | Parallel RunPod cloud sweep, result collection, pod cleanup |
| `extract_geometry.py` | One-time geometry extraction to a plain-tensor file |
| `multi_gpu_projector.py` | Ray-sharded multi-GPU projector wrapper |
| `geometry/geometry_80src_3rot_240view.pt` | Calibrated ring geometry (gitignored, 68 MB) |
| `results_archive/*.json` | Raw per-GPU results (meta + per-config rows) |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | Reproducible container env |
