"""EXPERIMENTAL: statistically weighted spectral preconditioner (3/4-exponent).

Approximates the inverse square root of the *weighted* Fisher operator

    K = A^T Sigma^{-1} A,        Sigma = diag(per-ray noise variance),

using only the reusable unweighted eigensystem of G = A^T A and the known
diagonal Sigma, at the cost of one forward + one backprojection per apply:

    N_W = F^{3/4} (A^T Sigma^{1/2} A) F^{3/4}
          + sigma_bar * tau^{-1/2} (I - V V^T)          ~=  K^{-1/2},

where F = V S^{-2} V^T + tau^{-1}(I - V V^T) is the null-rescaled inverse-Gram
approximation (tau = s_k^2, the smallest retained eigenvalue) and
F^{3/4} = V (S^{-3/2} - tau^{-3/4} I) V^T + tau^{-3/4} I is available in
closed form from the stored eigensystem.

Derivation (exponent matching in the SVD basis, exact for Sigma = sigma^2 I):
a symmetric sandwich F^a (A^T Sigma^gamma A) F^a has range-space spectrum
sigma^{2 gamma} s^{2-4a}; matching K^{-1/2} = sigma s^{-1} forces a = 3/4,
gamma = +1/2. Applied twice — exactly what
``bayesian_estimation.MaximumAPosterioriEstimator`` does with a
Preconditioner — it approximates K^{-1}, the Newton direction for the
weighted data term, so lr = 1.0 is the natural step size.

The complement term sigma_bar * tau^{-1/2} (I - V V^T) continues the
range-space spectrum sigma s^{-1} across the truncation edge (s_k, so
tau^{-1/2} = 1/s_k) and makes the operator symmetric positive definite,
hence invertible: a bijective image of the sufficient statistic A^T W b
stays sufficient. ``complement_variance`` selects the representative
variance sigma_bar^2: "median" continues the typical ray; "min" (the
smallest variance = largest weight) is the conservative choice that bounds
the complement spectral radius of N_W K N_W near 1 when the weights have a
large dynamic range.

This module exists for the one-off res64 weighted-preconditioning study; it
is not yet part of the production reconstruction path.
"""
import torch

from ..optimization import Preconditioner


class WeightedSpectralSqrtPreconditioner(Preconditioner):
    """N_W ~= (A^T Sigma^{-1} A)^{-1/2} from the unweighted eigensystem.

    Parameters
    ----------
    projector : object exposing ``forward_project`` / ``back_project`` and
        volume shape attributes ``n_x, n_y, n_z``.
    s : (k,) retained singular values of A, sorted descending and already
        condition-clipped (tau = s[-1]^2 is the truncation-edge eigenvalue).
    v : (N, k) retained image-domain eigenvectors.
    sigma : (n_ray,) per-ray noise VARIANCES  Sigma_ii = 1 / w_i.
    complement_variance : "median" | "min" | "mean" | float
        Policy for the representative variance sigma_bar^2 used on the
        unresolved complement (see module docstring).
    """

    def __init__(self, projector, s, v, sigma, complement_variance="median",
                 eps=1e-12):
        super().__init__()
        self.projector = projector
        self.nx, self.ny, self.nz = projector.n_x, projector.n_y, projector.n_z
        s = s.reshape(-1)
        if not bool((s[:-1] >= s[1:]).all()):
            raise ValueError("s must be sorted descending")
        self.register_buffer("v", v)
        tau = s[-1] ** 2
        # F^{3/4} spectral coefficients: s^{-3/2} on the range, tau^{-3/4} beyond
        self.register_buffer("f34_diag",
                             s.clamp(min=eps).pow(-1.5) - tau.pow(-0.75))
        self.register_buffer("f34_base", tau.pow(-0.75).reshape(()))
        self.register_buffer("sqrt_sigma", sigma.clamp(min=eps).sqrt())
        if complement_variance == "median":
            sig_bar2 = sigma.median()
        elif complement_variance == "min":
            sig_bar2 = sigma.min()
        elif complement_variance == "mean":
            sig_bar2 = sigma.mean()
        else:
            sig_bar2 = torch.as_tensor(float(complement_variance),
                                       device=sigma.device)
        # complement scale sigma_bar / s_k  (continuous with sigma s^{-1} at the edge)
        self.register_buffer("comp_scale",
                             (sig_bar2.sqrt() / tau.sqrt()).reshape(()))
        # inverse-operator coefficients: exponent matching for K^{1/2} gives
        # F^{1/4} (A^T Sigma^{-1/2} A) F^{1/4} + (1/sigma_bar) tau^{1/2} complement
        self.register_buffer("f14_diag",
                             s.clamp(min=eps).pow(-0.5) - tau.pow(-0.25))
        self.register_buffer("f14_base", tau.pow(-0.25).reshape(()))
        self.register_buffer("inv_comp_scale",
                             (tau.sqrt() / sig_bar2.sqrt()).reshape(()))

    # -- internals ----------------------------------------------------------
    def _fpow(self, x, diag, base):
        c = torch.mv(self.v.t(), x) * diag
        return torch.mv(self.v, c) + base * x

    def _sandwich(self, x, diag, base, ray_scale):
        """F^a-power  ->  A  ->  diag(ray_scale)  ->  A^T  ->  F^a-power."""
        y = self._fpow(x, diag, base)
        proj = self.projector.forward_project(
            y.reshape(self.nx, self.ny, self.nz).contiguous()).reshape(-1)
        proj = proj * ray_scale
        z = self.projector.back_project(proj).reshape(-1)
        return self._fpow(z, diag, base)

    def _complement(self, x):
        return x - torch.mv(self.v, torch.mv(self.v.t(), x))

    # -- Preconditioner contract --------------------------------------------
    def forward(self, x):
        xf = x.reshape(-1)
        out = self._sandwich(xf, self.f34_diag, self.f34_base, self.sqrt_sigma)
        return out + self.comp_scale * self._complement(xf)

    def inverse(self, x):
        """Approximate inverse ~= K^{1/2} (exact for Sigma proportional to I):
        F^{1/4} (A^T Sigma^{-1/2} A) F^{1/4} plus the matching complement."""
        xf = x.reshape(-1)
        out = self._sandwich(xf, self.f14_diag, self.f14_base,
                             1.0 / self.sqrt_sigma)
        return out + self.inv_comp_scale * self._complement(xf)
