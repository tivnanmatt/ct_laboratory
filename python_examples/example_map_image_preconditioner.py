#!/usr/bin/env python
"""MAP reconstruction with the IMAGE-domain sparse-eigen preconditioner.

End-to-end example of the ``ct_laboratory`` Bayesian-estimation stack:

    tomography   : a 3D cone-beam projector A  (forward_project / back_project)
    sparse_eigen_preconditioner
                 : SparseEigenDecomposition estimates the leading eigenpairs
                   (s_i^2, v_i) of the Gram operator G = A^T A, then
                   ``to_image_preconditioner()`` returns a
                   SparseEigenImagePreconditioner
                       P = V (S^-1 - s_min^-1 I) V^T + s_min^-1 I
                   which stores the image eigenvectors V (N x k) and acts on the
                   volume directly (two matmuls per apply, no projector inside).
    random_variable
                 : likelihood  y | x ~ N(A x, sigma^2 I)   (LinearGaussian)
                   prior       p(x) ∝ exp(-(w/2) ||grad x||^2)  (smoothness)
    bayesian_estimation
                 : MaximumAPosterioriEstimator minimizes
                       -log p(y|x) - log p(x)
                   by preconditioned gradient descent. It applies the
                   preconditioner as a linear operator to the gradient
                   (direction = F g = P(P(g)), F the approximate inverse
                   Hessian), so the estimate converges in a handful of steps
                   where unpreconditioned gradient descent crawls.

Companion example ``example_map_projection_preconditioner.py`` swaps in the
PROJECTION-domain preconditioner (same operator, stores p = A V and reaches the
volume through the projector) and is otherwise identical.

Runs on GPU if available, else CPU (torch backend). Writes a comparison montage
to ``python_examples/test_outputs/map_image_preconditioner.png``.
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
    """A simple 3D block phantom (two nested boxes) in attenuation units."""
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

    # --- geometry + projector (small, so it runs fast on CPU) -----------------
    nx, ny, nz = 32, 32, 8
    proj = ConeBeam3DProjector(
        n_x=nx, n_y=ny, n_z=nz, n_view=48, det_nx=48, det_ny=12,
        sid=200.0, sdd=400.0, voxel_size=1.0, backend=backend, device=device)

    # --- forward model + simulated measurement --------------------------------
    x_true = build_phantom(nx, ny, nz, device)
    noise_std = 0.1
    y_clean = proj.forward_project(x_true)
    y = y_clean + noise_std * torch.randn_like(y_clean)

    # --- offline: leading eigenpairs of G = A^T A -> image preconditioner -----
    t0 = time.time()
    dec = SparseEigenDecomposition(proj, k=48)
    dec.compute_weights(method="subspace", n_iters=25, use_tqdm=False)
    P_image = dec.to_image_preconditioner()          # stores V (N x k)
    print(f"eigendecomposition (k={dec.s.numel()}) in {time.time()-t0:.1f}s; "
          f"singular values in [{dec.s.min():.2f}, {dec.s.max():.2f}], "
          f"Gram condition ~ {(dec.s.max()/dec.s.min())**2:.1f}")

    # --- probabilistic model: Gaussian likelihood + smoothness prior ----------
    likelihood = LinearGaussianRandomVariable(
        op=lambda vol: proj.forward_project(vol.reshape(nx, ny, nz)),
        var=noise_std ** 2)
    prior = QuadraticSmoothnessRandomVariable(weight=1e-2)

    # --- MAP with the image-domain preconditioner -----------------------------
    # The preconditioner F = P^2 approximates (A^T A)^{-1}; the Gaussian data
    # Hessian is (1/var) A^T A, so lr = var makes one preconditioned step the
    # NEWTON step for the data term (the exact minimizer along each modeled
    # eigen-direction). That is why preconditioned MAP converges in a handful of
    # iterations.
    def data_misfit(vol):
        return (proj.forward_project(vol.reshape(nx, ny, nz)) - y).norm().item() \
            / y.norm().item()

    x0 = torch.zeros(nx, ny, nz, device=device)
    est = MaximumAPosterioriEstimator(
        likelihood, prior, measurements=y, x_init=x0,
        preconditioner=P_image, lr=noise_std ** 2)
    t0 = time.time()
    x_precond = est.estimate(num_iters=40, use_tqdm=False)
    print(f"preconditioned MAP: 40 iters in {time.time()-t0:.2f}s  "
          f"RMSE={rmse(x_precond.reshape(nx, ny, nz), x_true):.3f}  "
          f"data-misfit={data_misfit(x_precond):.3f}")

    # --- baseline: same objective, NO preconditioner --------------------------
    # Unpreconditioned gradient descent is limited by the largest Gram
    # eigenvalue: its stable step is ~2*var/s_max^2 (here ~4e-6), so in the same
    # 40 iterations it makes far less progress. It needs ~500 iterations to
    # reach the misfit the preconditioner reaches in 40.
    est_gd = MaximumAPosterioriEstimator(
        likelihood, prior, measurements=y, x_init=x0.clone(),
        preconditioner=None, lr=2e-6)
    x_gd = est_gd.estimate(num_iters=40, use_tqdm=False)
    print(f"plain gradient descent: 40 iters  "
          f"RMSE={rmse(x_gd.reshape(nx, ny, nz), x_true):.3f}  "
          f"data-misfit={data_misfit(x_gd):.3f}")
    print(f"(reference: all-zeros RMSE = {rmse(x0, x_true):.3f}; the RMSE floor "
          f"is set by the {dec.s.numel()} modeled modes, not the noise)")

    # --- montage: center axial slice of truth / precond MAP / plain GD --------
    z = nz // 2
    vols = [("ground truth", x_true),
            ("MAP + image preconditioner", x_precond.reshape(nx, ny, nz)),
            ("plain gradient descent", x_gd.reshape(nx, ny, nz))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    vmin, vmax = 0.0, float(x_true.max())
    for ax, (title, vol) in zip(axes, vols):
        ax.imshow(vol[:, :, z].detach().cpu(), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MAP reconstruction with sparse-eigen IMAGE preconditioner "
                 f"(axial z={z}, 40 iterations each)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "map_image_preconditioner.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
