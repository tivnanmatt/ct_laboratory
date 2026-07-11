# ct_laboratory

A differentiable CT reconstruction library built on PyTorch with custom CUDA
kernels. The core is an intersection-based (Siddon-type) forward/backprojector
for 2D and 3D geometries, exposed as autograd-compatible `nn.Module`s so that
reconstruction can be written as gradient-based optimization. On top of the
projectors the package provides scanner geometry classes (fan-beam, cone-beam,
and multi-source **StaticCT** ring systems), matrix-free sparse
eigendecomposition of the projector Gram operator, MAP/Bayesian reconstruction
drivers with pluggable likelihoods, priors, and preconditioners, a spectral
X-ray physics package, and a cross-GPU performance benchmark suite.

## Package areas

The package is organized into **areas** with a strict downward dependency
order — base areas depend on nothing else in the package, composite areas
combine them:

| Area | Depends on | Contents |
|------|-----------|----------|
| `ct_laboratory.tomography` | — (base) | All projectors: 2D/3D torch/cuda/autograd/module layers, fan-beam, cone-beam, StaticCT ring (2D/3D), multi-rotation wrapper, ray-subset projectors, voxel→world transforms |
| `ct_laboratory.optimization` | — (base) | `Preconditioner` abstract base class (subclasses MUST implement both `forward` and `inverse` — enforced at class definition), `IdentityPreconditioner`, `DiagonalPreconditioner`, general loss functions (`LossFunction`, `WeightedSumLoss`, `QuadraticLoss`) |
| `ct_laboratory.random_variable` | — (base) | `RandomVariable` / `ConditionalRandomVariable` on top of `torch.distributions` (`log_prob`/`log_pdf`, `sample`, `score`); Gaussian, Poisson/Beer–Lambert, energy-based smoothness/TV variables, `FromTorchDistribution` bridge |
| `ct_laboratory.sparse_eigen_preconditioner` | optimization + tomography | `SparseEigenDecomposition` (matrix-free eigenpairs of G = AᵀA) and the image/projection sparse-eigen preconditioners, subclassing `optimization.Preconditioner` |
| `ct_laboratory.bayesian_estimation` | random_variable + optimization | `MaximumAPosterioriEstimator`: likelihood = a `ConditionalRandomVariable`, prior = an unconditional `RandomVariable` (no separate likelihood/prior aliases), preconditioning via a single `Preconditioner` object |
| `ct_laboratory.physics.xray` | — (no projector deps) | Spectral X-ray measurement physics operating purely on basis line integrals |
| `ct_laboratory.physics.ct_system` | tomography + physics.xray | `CTSystem`: projector + `XraySystem` composed into one differentiable spectral scan model |
| `ct_laboratory.linalg` | — (base) | Consolidated matrix-free `LinearSystem` stack (merged from gmi): `Identity`, `Scalar`, `DiagonalScalar`, `FourierTransform`/`FourierConvolution`, composites, conjugate/transpose/inverse wrappers, SVD/eigen-decomposed operators, bilinear/Lanczos/nearest interpolators, polar resampler, sparse row/col systems |
| `ct_laboratory.linear_system` | — (base) | Legacy multi-file `LinearSystem` stack (merged from gmi); parallel to `linalg`, kept because `sde` targets it. One class per file (`base`, `composite`, `scalar`, `identity`, `fourier_*`, …) plus vendored hydra configs for the test suite |
| `ct_laboratory.sde` | linear_system | Stochastic differential equations (merged from gmi): `StochasticDifferentialEquation`, `LinearSDE`, `ScalarSDE`, `VarianceExploding/PreservingSDE`, `StandardWienerSDE`, `Diagonal/FourierSDE` |
| `ct_laboratory.diffusion` | sde + linear_system + random_variable_gmi + samplers | Diffusion models (merged from gmi): `DiffusionModel`, `DiffusionPosteriorModel`, reverse-process + DPS samplers. Heavy deps (wandb, torch_ema, hydra) load lazily; the gmi `datasets` area was **not** merged, so real-data training/sampling needs areas outside this repo |
| `ct_laboratory.random_variable_gmi` | linalg + linear_system + sde + samplers | gmi's random-variable stack (Gaussian/Uniform/LogNormal/Categorical + measurement simulator). Renamed from gmi's `random_variable` to avoid colliding with the native `ct_laboratory.random_variable` |
| `ct_laboratory.samplers`, `.config`, `.train` | — | Support modules pulled in by the diffusion stack (base `Sampler`, config object loader, training loop). `train` imports `torch_ema` only when used |

The legacy flat namespace is preserved: every pre-existing import path
(`ct_laboratory.ct_projector_3d_module`, `ct_laboratory.map_reconstructor`,
`ct_laboratory.sparse_eigen_decomposition`, ...) still works via
backward-compatibility shims that alias the new area modules.

## Repository layout

```
ct_laboratory/
├── ct_laboratory/            The Python package
│   ├── _C.*.so               Compiled CUDA extension (built from src/)
│   ├── tomography/           BASE AREA: all projectors + grid transforms
│   ├── optimization/         BASE AREA: Preconditioner contract + losses
│   ├── random_variable/      BASE AREA: RandomVariable interface over
│   │                         torch.distributions (priors & likelihoods)
│   ├── sparse_eigen_preconditioner/  Gram eigenpairs + preconditioners
│   ├── bayesian_estimation/  MAP estimation on random variables
│   ├── physics/
│   │   ├── xray/             Spectral X-ray physics (projector-free)
│   │   └── ct_system/        Projector + XraySystem integration
│   ├── map_reconstructor.py  Legacy MAP driver (kept for compatibility)
│   ├── *_likelihood.py, *_prior.py   Legacy likelihood/prior modules
│   ├── bayesian_diffusion_posterior_sampling.py  DPS reconstruction driver
│   └── ct_projector_*.py, staticct_*.py, ...     Compatibility shims →
│                                                  tomography/
├── src/                      CUDA/C++ extension source (bindings + kernels)
├── benchmark/                Cross-GPU projector timing suite (see its README)
├── python_examples/          Runnable demos and tests (write to test_outputs/)
├── setup.py, Makefile        Extension build and package install
├── build/, ct_laboratory.egg-info/, outputs/   Generated (not source)
```

## The projector core

Every projector is built on the same primitive: for each ray (a `src`/`dst`
endpoint pair in world coordinates) the intersection parameters t ∈ [0, 1]
with the voxel grid planes are computed, sorted, and used to accumulate
piecewise-constant line integrals. Forward projection sums voxel values times
segment lengths; backprojection is its exact adjoint (scatter-add). Custom
`torch.autograd.Function`s wire the pair together, so `forward_project` is
differentiable and `loss.backward()` performs a backprojection — iterative
reconstruction falls out of standard PyTorch optimization.

**Layers per dimension (2D and 3D mirror each other):**

| Module | Role |
|---|---|
| `ct_projector_*_torch.py` | Pure-PyTorch reference implementation (any device, slow) |
| `ct_projector_*_cuda.py` | Thin wrappers over the compiled `ct_laboratory._C` kernels |
| `ct_projector_*_autograd.py` | `autograd.Function`s pairing forward/backprojection |
| `ct_projector_*_module.py` | `CTProjector2DModule` / `CTProjector3DModule`: the user-facing `nn.Module`; precomputes intersections at construction, `backend='cuda' | 'torch'` |

**Three intersection-weight strategies (3D).** The 3D module supports three
ways to obtain the per-ray intersection weights ("tvals"), trading memory for
speed:

1. **Precomputed float32** — fastest to build from, largest memory.
2. **Precomputed uint16-compressed** (`use_compression=True`, default in
   production) — intersection params packed to uint16 deltas
   (`compress_tvals_to_uint16` / `compress_tvals_3d_cuda`); half the memory of
   float32 and equal or better speed (the operator is memory-bandwidth-bound).
   `precompute_tvals_stitched(...)` builds these in ray chunks for arbitrarily
   large geometries, and the resulting tensors can be saved to disk and passed
   back via the `tvals=` constructor argument — a ~100× per-application
   speedup over on-the-fly recomputation for a one-time build cost
   (see `benchmark/README.md`).
3. **On-the-fly Siddon** — no stored weights; intersections recomputed per
   call. Slowest per application, zero resident weight memory; also the
   automatic fallback when no tvals are supplied.

**Volume ↔ world mapping.** All projectors take an affine map `(M, b)` from
voxel index (i, j, k) to world (x, y, z); `standard_image_transform_2d/3d`
builds the common centered-volume diagonal-spacing case. The CUDA kernels
receive `M⁻¹` (prepared by the wrappers).

## Geometry classes

- **`FanBeam2DProjector`** (`fanbeam_projector_2d.py`) — standard rotating
  fan-beam from `n_view`, `n_det`, SID/SDD, detector spacing;
  `build_fanbeam_2d_geometry` generates the src/dst rays.
- **`ConeBeam3DProjector`** (`conebeam_projector_3d.py`) — circular cone-beam
  with a flat detector; `build_conebeam_3d_geometry` vectorizes all
  `n_view × det_ny × det_nx` rays.
- **`StaticCTProjector2D/3D`** (`staticct_projector_*.py`) — the general
  multi-source, multi-module ring scanner with NO gantry rotation: arbitrary
  source positions, detector module centers/orientations, a
  `source_module_mask` selecting which modules are read out per source, per-
  frame gantry transforms `(M_gantry, b_gantry)`, and an `active_sources`
  schedule. Frame-sequence builders `build_circular_sequence` /
  `build_helical_sequence` generate rotation/helical firing patterns.
- **`UniformStaticCTProjector2D/3D`** — parametric convenience subclasses that
  generate the general geometry from scalars (`n_source`, `source_radius`,
  `n_module`, `module_radius`, detector pixels/spacings per module,
  `modules_per_source`, z offsets) via `build_uniform_static_*_geometry`, then
  defer to the general base class.
- **`MultiRotationProjector`** (`ct_projector_3d_multirot.py`) — rolling-window
  multi-rotation wrapper for helical-style scans: slices the relevant z-window
  of a long volume per rotation and projects it through a base projector.
- **Subset projectors** (`ct_projector_3d_subsets.py`) —
  `OrderedSubsetProjector` (blocks of at most `max_subset_size` rays, optional
  on-disk tvals cache) and `RandomSubsetProjector` (random ray subsets,
  optionally re-randomized per forward/backward) for ordered-subset and
  stochastic reconstruction; both wrap any `CTProjector3DModule`-derived
  projector.

## Sparse eigendecomposition (`sparse_eigen_decomposition.py`)

`SparseEigenDecomposition(operator, k)` estimates the leading k eigenpairs of
the Gram operator **G = AᵀA without ever materializing it** — every Gram
product is one forward + one backprojection on GPU. It stores `s = √eigenvalue`
(k,), image-domain eigenvectors `v` (N, k), and their projections `p = A v`
(n_ray, k) as buffers, with `save`/`load` for reuse. Two registered solvers,
selected by `compute_weights(method=...)`:

- `"subspace"` (default) — block power iteration + Rayleigh–Ritz;
  dependency-free.
- `"eigsh"` — SciPy ARPACK (`eigsh(which='LM')`) driven through a
  `LinearOperator`; more accurate spectra for production bases.

New solvers can be plugged in via `SparseEigenDecomposition.register_method`.
These eigenpairs are the weights behind the sparse-eigen preconditioners and
the multi-resolution reconstruction pipeline in `research-ring`.

## MAP / Bayesian reconstruction stack

Reconstruction is posed as maximum a posteriori estimation: minimize
−(log likelihood + log prior) over the volume. The pieces compose freely:

**Likelihoods** — `LinearGaussianLogLikelihood` /
`DiagonalGaussianLogLikelihood` (Gaussian data terms through a linear
projector) and `NonlinearPoissonLogLikelihood` (Beer–Lambert photon counting,
counts = I₀·exp(−Ax), for realistic low-dose statistics).

**Priors** — `DiagonalGaussianLogPrior` (element-wise Gaussian),
`QuadraticSmoothnessLogPrior2D/3D` (Laplacian ‖∇x‖² smoothness),
`TotalVariancePrior2D/3D` (edge-preserving TV), and `BayesianDenoiserPrior`
(plug-and-play prior wrapping a learned denoiser with a noise-variance
schedule).

**Preconditioners** (`map_reconstructor.py`) — `SparseEigenImagePreconditioner`
and `SparseEigenProjectionPreconditioner` implement
P = V(S⁻¹ − s_min⁻¹ I)Vᵀ + s_min⁻¹ I from a saved `SparseEigenDecomposition`:
the top-k modes are rescaled by their singular values and every unmodeled
direction by s_min⁻¹, which flattens the Gram spectrum and dramatically
accelerates gradient-based reconstruction. The image variant applies V
directly; the projection variant reaches the volume through A/Aᵀ using the
stored `p = AV`.

**Drivers** — `MaximumAPosterioriReconstructor` (SGD on the optionally
preconditioned volume, LR warmup, subset re-randomization hooks; `map_step` /
`map_reconstruction`) and `BayesianDiffusionPosteriorSampling` (diffusion
posterior sampling: alternates denoiser-prior noise levels with MAP inner
loops).

## Spectral X-ray physics (`ct_laboratory/physics/xray/`)

A composable, differentiable spectral CT forward model. Submodules:

- `materials/` — `MaterialDatabase` (elemental/compound attenuation data),
  energy-dependent μ utilities (`get_mu`, photoelectric/Compton splits), and
  basis-material parameterizations (`BasisMaterials`, `PECSBasis`,
  `WaterIodineBasis`, `WaterCalciumBasis`, `WaterBasis`, `SoftTissueBasis`).
- `source/` — spectrum models: `SpekpyXraySource` (spekpy-generated tube
  spectra), `MonoenergeticXraySource`, aluminium/material-filtered variants,
  and `DualExposureXraySource` for kV-switching studies.
- `attenuation/` — operators applying basis line integrals to spectra
  (`BasisAttenuator`, `ObjectAttenuator`) and fixed filtration
  (`UniformAluminumFilter`, `MaterialFilter`, `RayDependentFilter`).
- `detector/` — `EnergyIntegratingCTDetector`, `DualExposureCTDetector`,
  GOS interaction models, and detector-side blur.
- `blur/`, `scatter/` — projection-domain Gaussian blur (invertible Fourier
  implementation) and zero/constant scatter models.
- `optimization/` — `NewtonOptimizer`, a second-order solver for inverting the
  spectral model (e.g. material decomposition).
- `xray_system/` — `XraySystem`, the end-to-end chain
  source → filtration → object attenuation → blur → interaction → detector,
  with Poisson statistics and consistent tensor shapes
  (`x_basis: [n_rays, n_materials]`, `y: [n_rays, n_channels]`).

## Merged generative stack (from gmi)

The linear-algebra, SDE, and diffusion areas were merged in from the
**Generative Medical Imaging (gmi)** package to make its matrix-free operators
and diffusion models usable alongside the CT projectors. All are pure-PyTorch
(no CUDA extension required) and run on CPU or GPU.

- **`linalg`** and **`linear_system`** are two parallel `LinearSystem`
  hierarchies that both came from gmi (a consolidated single-package version
  and the older one-class-per-file version). Both are kept: `linalg` is the
  richer/newer API; `linear_system` is retained because `sde` imports from it.
  A `LinearSystem` exposes `forward`/`transpose`/`conjugate`/`inverse` and
  composes (`CompositeLinearSystem`, transpose/conjugate/inverse wrappers).
- **`sde`** builds forward/reverse stochastic processes on top of
  `linear_system` operators (the diffusion term of an SDE returns a
  `LinearSystem`).
- **`diffusion`** composes an SDE, a backbone `nn.Module`, a training-time
  sampler, and a loss into a `DiffusionModel` with training-loss and
  reverse-process/DPS sampling. The gmi `datasets` area was **deliberately not
  merged**, so the built-in data-loading training entry points require modules
  that live outside this repo; use the dataset-free path (supply your own
  tensors and backbone) for standalone use.

What was intentionally left in gmi: `datasets`, `network`, `models`, `tasks`,
`loss_function`, and the CLI. The `random_variable` area was renamed to
`random_variable_gmi` here to avoid colliding with the native
`ct_laboratory.random_variable`.

Regression tests for the whole merged stack live in
`python_examples/gmi_merge_tests/` (see below).

## CUDA extension (`src/`)

`bindings.cpp` (pybind11) builds the `ct_laboratory._C` module exposing the
kernel families implemented in `ct_projector_2d.cu` / `ct_projector_3d.cu`:
intersection precompute (`compute_intersections_2d/3d`), precomputed-weight
forward/backprojection, uint16-compressed variants
(`compress_tvals_3d_cuda`, `forward/back_project_3d_compressed_cuda`), and
on-the-fly Siddon paths. Backprojection uses `atomicAdd` scatter;
a `ray_kernel_plane_sort` helper sorts intersection parameters per ray.

## Build and install

```bash
make build     # compile the CUDA extension (torch.utils.cpp_extension, ninja)
make develop   # editable install (--user)
make install   # regular install (--user)
make clean     # remove build/, dist/, egg-info and pip-uninstall
```

Requirements: PyTorch with CUDA, `nvcc` on PATH (the Makefile auto-detects
`CUDA_HOME`), NumPy/Matplotlib for the examples, SciPy for the `eigsh`
eigensolver, and spekpy only if `SpekpyXraySource` is used. `setup.py` builds
the single `CUDAExtension` from the three files in `src/`.

## Examples (`python_examples/`)

All scripts run standalone and write PNGs/MP4s to
`python_examples/test_outputs/`:

- **Line-integral / module tests** — `test_ct_projector_{2d,3d}_line_integral.py`,
  `test_ct_projector_{2d,3d}_module.py`: low-level forward/back checks and
  torch-vs-CUDA backend parity.
- **Fan-beam / cone-beam** — `test_*fanbeam*` and `test_*conebeam*`: sinogram
  generation and autograd-based iterative reconstruction
  (`*_autorecon.py` produce MP4 animations of the recon converging).
- **StaticCT** — `test_staticct_projector_2d.py` / `_3d.py` (geometry and
  sinogram visualization), `*_autorecon.py` (iterative recon; the 2D version
  reconstructs a DICOM phantom), and
  `test_staticct_projector_2d_nsource_sweep.py` (image quality vs number of
  sources).
- **Timing** — `timing_sparse_eig_k64.py`: `SparseEigenDecomposition` k=64 on
  the uniform StaticCT 3D projector.
- **Merged-stack tests** — `gmi_merge_tests/` is a pytest suite (313 ported
  from gmi + 6 new) covering `linalg`, `linear_system`, `sde`, and dataset-free
  `diffusion` smoke tests. Runs on CPU:
  `PYTHONPATH=. python -m pytest python_examples/gmi_merge_tests -q`
  (319 passing).

## Benchmark suite (`benchmark/`)

Cross-GPU forward/backprojection timing on the full calibrated StaticCT
geometry (240 views, 2,949,120 rays): 17+ GPU configurations from RTX 3090 to
H100, three weight strategies, a RunPod cloud sweep orchestrator, and a
ray-sharded multi-GPU wrapper (`multi_gpu_projector.py`) that the production
reconstruction pipeline uses to hold res512 on 2×24 GB cards. Headline: one
512×512×64 projector application costs ~46–72 ms on modern GPUs with uint16
precomputed weights, and ray sharding reaches 21/28 ms on 4× L40S. Full
motivation, methodology, results tables, and the memory strategy are in
[benchmark/README.md](benchmark/README.md).

## Generated directories

`build/`, `dist/`, `ct_laboratory.egg-info/` are setuptools artifacts;
`outputs/` and `python_examples/test_outputs/` hold script outputs. None are
source; all are safe to delete and regenerate.
