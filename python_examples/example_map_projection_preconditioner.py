#!/usr/bin/env python
"""MAP reconstruction with the PROJECTION-domain sparse-eigen preconditioner.

Identical to ``example_map_image_preconditioner.py`` except for how the
preconditioner is built. Both implement the SAME operator
    P = V (S^-1 - s_min^-1 I) V^T + s_min^-1 I,
but through different computational pathways:

    IMAGE      SparseEigenImagePreconditioner(s, v)
               stores the image eigenvectors V (N x k); two matmuls per apply;
               no projector involved.

    PROJECTION SparseEigenProjectionPreconditioner(projector, p, s2)
               stores the projection eigenvectors p = A V (n_ray x k) and reaches
               the volume ONLY through the projector A / A^T. The projector
               instance lives inside the preconditioner. Each apply therefore
               costs one forward + one back projection.

Which to use:
  * If V fits in memory (moderate k), prefer the IMAGE preconditioner — it is
    cheaper per iteration.
  * Use the PROJECTION preconditioner when storing p is preferable to storing V
    (or V is too large), accepting the extra fwd/back projection per apply.

Note the parameterization difference: the image variant takes singular values
``s``; the projection variant takes eigenvalues ``s2 = s**2``. The helper
``dec.to_projection_preconditioner()`` handles this and wires in the projector
that produced the basis.

Because ``MaximumAPosterioriEstimator`` applies the preconditioner as a linear
operator to the gradient (it does NOT differentiate through it), the projection
preconditioner's internal back_project call — a raw kernel, not an autograd
Function — is used safely.

Writes ``python_examples/test_outputs/map_projection_preconditioner.png``.
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from ct_laboratory.tomography import ConeBeam3DProjector
from ct_laboratory.sparse_eigen_preconditioner import SparseEigenDecomposition
from ct_laboratory.random_variable import (LinearGaussianRandomVariable,
                                           QuadraticSmoothnessRandomVariable)
from ct_laboratory.bayesian_estimation import MaximumAPosterioriEstimator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "test_outputs")


def build_phantom(nx, ny, nz, device):
    x = torch.zeros(nx, ny, nz, device=device)
    x[nx // 4:3 * nx // 4, ny // 4:3 * ny // 4, nz // 4:3 * nz // 4] = 1.0
    x[3 * nx // 8:5 * nx // 8, 3 * ny // 8:5 * ny // 8, :] = 2.0
    return x


def rmse(a, b):
    return (a - b).pow(2).mean().sqrt().item()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "cuda" if device.type == "cuda" else "torch"
    print(f"device={device}  backend={backend}")

    nx, ny, nz = 32, 32, 8
    proj = ConeBeam3DProjector(
        n_x=nx, n_y=ny, n_z=nz, n_view=48, det_nx=48, det_ny=12,
        sid=200.0, sdd=400.0, voxel_size=1.0, backend=backend, device=device)

    x_true = build_phantom(nx, ny, nz, device)
    noise_std = 0.1
    y_clean = proj.forward_project(x_true)
    y = y_clean + noise_std * torch.randn_like(y_clean)

    # --- eigendecomposition; compute_weights also stores p = A V (needed here) -
    t0 = time.time()
    dec = SparseEigenDecomposition(proj, k=48)
    dec.compute_weights(method="subspace", n_iters=25, use_tqdm=False)
    # the projector that produced the basis is wired in automatically; you could
    # also pass one explicitly: dec.to_projection_preconditioner(projector=proj)
    P_projection = dec.to_projection_preconditioner()
    print(f"eigendecomposition (k={dec.s.numel()}) in {time.time()-t0:.1f}s; "
          f"projection basis p shape = {tuple(dec.p.shape)}")

    likelihood = LinearGaussianRandomVariable(
        op=lambda vol: proj.forward_project(vol.reshape(nx, ny, nz)),
        var=noise_std ** 2)
    prior = QuadraticSmoothnessRandomVariable(weight=1e-2)

    # lr = var makes one preconditioned step the Newton step for the Gaussian
    # data term (F approximates (A^T A)^{-1}; the data Hessian is (1/var) A^T A).
    def data_misfit(vol):
        return (proj.forward_project(vol.reshape(nx, ny, nz)) - y).norm().item() \
            / y.norm().item()

    x0 = torch.zeros(nx, ny, nz, device=device)
    est = MaximumAPosterioriEstimator(
        likelihood, prior, measurements=y, x_init=x0,
        preconditioner=P_projection, lr=noise_std ** 2)
    t0 = time.time()
    x_precond = est.estimate(num_iters=40, use_tqdm=False)
    print(f"preconditioned MAP: 40 iters in {time.time()-t0:.2f}s  "
          f"RMSE={rmse(x_precond.reshape(nx, ny, nz), x_true):.3f}  "
          f"data-misfit={data_misfit(x_precond):.3f}")

    # unpreconditioned baseline: stable step ~2*var/s_max^2, far slower.
    est_gd = MaximumAPosterioriEstimator(
        likelihood, prior, measurements=y, x_init=x0.clone(),
        preconditioner=None, lr=2e-6)
    x_gd = est_gd.estimate(num_iters=40, use_tqdm=False)
    print(f"plain gradient descent: 40 iters  "
          f"RMSE={rmse(x_gd.reshape(nx, ny, nz), x_true):.3f}  "
          f"data-misfit={data_misfit(x_gd):.3f}")

    z = nz // 2
    vols = [("ground truth", x_true),
            ("MAP + projection preconditioner", x_precond.reshape(nx, ny, nz)),
            ("plain gradient descent", x_gd.reshape(nx, ny, nz))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    vmin, vmax = 0.0, float(x_true.max())
    for ax, (title, vol) in zip(axes, vols):
        ax.imshow(vol[:, :, z].detach().cpu(), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MAP reconstruction with sparse-eigen PROJECTION preconditioner "
                 f"(axial z={z}, 40 iterations each)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "map_projection_preconditioner.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
