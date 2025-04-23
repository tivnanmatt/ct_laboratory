# README

This repository contains a Python-based CT reconstruction library and related test scripts. The code implements both 2D and 3D tomographic projectors in PyTorch, with optional CUDA-accelerated kernels for intersection-based forward and back projection. It also includes scripts and utilities for running fan-beam, cone-beam, and static 3D CT simulations (including some helical trajectories). Below is an overview of the directory structure and main components.

## Directory Structure
```
ct_laboratory
├── .vscode/
│   └── settings.json
├── Makefile
├── bin/
├── build/
│   ├── lib.linux-x86_64-cpython-312/
│   │   └── ct_laboratory/
│   │       (...compiled extension modules and Python sources...)
│   └── temp.linux-x86_64-cpython-312/
│       (...ninja build artifacts...)
├── dist/
├── env_notes.txt
├── gmm3d_prior.pth
├── gmm_regularizer3d.pth
├── include/
├── moons.py
├── out/
├── print_code.py
├── python_package/
│   ├── ct_laboratory/
│   │   (...Python sources for CT projector modules...)
│   ├── ct_laboratory.egg-info/
│   └── setup.py
├── python_test/
│   (...unit tests and demonstration scripts...)
├── reverse_process_and_score_animation.mp4
├── score_based_training_animation.mp4
├── src/
│   ├── bindings.cpp
│   ├── ct_projector_2d.cu
│   └── ct_projector_3d.cu
├── static_ct_simulator/
│   (...example scripts for static/helical CT reconstruction, GMM priors, etc.)
├── tex/
│   └── ct_projector_2d.tex
├── tmp.py
└── validation.png
```

## Key Highlights

### Makefile
- Builds and installs the C++/CUDA extension via `python setup.py build`, `develop`, or `install`.
- `make build` compiles the CUDA extension into `build/`.
- `make develop` or `make install` registers the Python package.

### src/
- Contains the main C++/CUDA source files (.cu, .cpp) that implement intersection-based forward/back projectors in 2D and 3D.
- `bindings.cpp` exposes these functions to PyTorch via PyBind11.
- `ct_projector_2d.cu` and `ct_projector_3d.cu` implement CUDA kernels for computing intersection parameters and performing forward/back projection.

### python_package/
- `setup.py`: Uses `torch.utils.cpp_extension` to build the CUDA extension.
- `ct_laboratory/`: The Python package code:
  - `ct_projector_2d_*.py` and `ct_projector_3d_*.py`: PyTorch-based and CUDA-based projectors (autograd-compatible).
  - `fanbeam_projector_2d.py`, `conebeam_projector_3d.py`: High-level classes for standard fan-beam or cone-beam geometries.
  - `staticct_projector_2d.py`, `staticct_projector_3d.py`: More advanced classes for “static” CT systems with complex geometry, modules, frames, etc.

### python_test/
- Contains numerous test scripts and demonstration code:
  - `test_ct_projector_2d_*.py` and `test_ct_projector_3d_*.py`: Basic unit tests for 2D/3D forward/back projection, including line integral checks.
  - `test_fanbeam_projector_2d.py`, `test_conebeam_projector_3d.py`: End-to-end fan-beam/cone-beam forward projection demos.
  - `test_*_autorecon.py`: Autograd-based reconstruction examples (iterative gradient descent with MSE loss).
  - Several scripts produce animations illustrating iterative reconstruction.

### static_ct_simulator/
- `mtec_system.py`, `arpa_system_multirot.py`, etc.: Code for specialized “MTEC” or “ARPA” geometry, multi-rotation/helical scanning.
- `gmm_prior.py`: An example patch-based GMM prior for 3D volumes, trained via random patch sampling.
- Demonstrates how to shift volumes or frames for simulating helical motion, combining the static projector in multiple passes.

### moons.py
- Contains a 2D “score-based” generative modeling example (unrelated to CT, but included as a demonstration of training loops, animation, etc.).

### print_code.py
- A utility script that prints out the directory structure and source files.

### Various .mp4 animations
- `score_based_training_animation.mp4`, `reverse_process_and_score_animation.mp4`: From the score-based diffusion code (`moons.py`).
- `auto_recon_*.mp4`: Iterative recon demos from the test scripts.

## Usage

### Setup / Build
1. Ensure you have PyTorch with CUDA and the usual Python environment.
2. Run `make build` to compile the CUDA extension into `build/`.
3. Run `make develop` (or `make install`) to install the package locally.

### Running Tests
From the `ct_laboratory/` root, execute scripts in `python_test/` to test various functionalities, for example:
```bash
cd python_test
python test_ct_projector_2d_autorecon.py
python test_conebeam_projector_3d_autorecon.py
```
These scripts will produce sinograms, reconstructions, and possibly MP4 animations in their `test_outputs/` subfolder.

## Examples
- **2D Fan-Beam:** `python_test/test_fanbeam_projector_2d.py` or `_autorecon.py`
  - Demonstrates a fan-beam geometry and iterative reconstruction using PyTorch autograd.
- **3D Cone-Beam:** `python_test/test_conebeam_projector_3d.py` or `_autorecon.py`
  - Shows a standard circular cone-beam setup with a simple spherical phantom.
- **Static / Helical CT:** Scripts in `static_ct_simulator/` show more complex geometry with modules, frames, ring-based or helical scanning, and GMM regularization examples.

## Extending
- To modify the CUDA kernels, edit `src/ct_projector_*.cu`, then rebuild with `make build`.
- New geometry classes can be added similarly to `fanbeam_projector_2d.py` or `conebeam_projector_3d.py`.

## Code Overview

### Intersection Computation (`ct_projector_2d.cu`, `ct_projector_3d.cu`)
- Each ray’s intersection parameters with the discrete image/volume grid are computed and sorted in [0,1].

### Forward Projection
- Segments each ray in piecewise-constant steps, summing pixel/voxel values multiplied by segment lengths.

### Back Projection
- Distributes sinogram intensities back into the image/volume via `atomicAdd`.

### PyTorch Integration
- Python-side modules (e.g., `ct_projector_2d_module.py`, `ct_projector_3d_module.py`) precompute intersection parameters once.
- Custom autograd Functions handle forward/back pass to enable gradient-based reconstruction.

### High-Level Classes
- `FanBeam2DProjector`, `ConeBeam3DProjector`: Standard parametric fan/cone geometry.
- `StaticCTProjector2D/3D`: Flexible “static” geometry with modules, frames, arbitrary source positions, etc.

### Scripts
- Many scripts under `python_test/` do demonstration or unit testing.
- `static_ct_simulator/` has more advanced multi-rotation / helical examples (e.g., shifting volumes, GMM-based priors, etc.).

## License and Contributions
- **License:** (Add your preferred license here, e.g., MIT, Apache 2.0)
- **Contributions:** Pull requests are welcome. Please open issues for bug reports or feature requests.

## Contact
For questions or discussions, feel free to open an issue on GitHub or contact the maintainers directly.

