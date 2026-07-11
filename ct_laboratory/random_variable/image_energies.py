"""Energy-based (improper) image random variables: smoothness and TV.

These are Gibbs variables p(x) ∝ exp(−w·E(x)) whose normalizing constant is
never needed for MAP estimation or scores; ``normalized`` is False. The
finite-difference energies mirror the legacy ``quadratic_smoothness_prior`` /
``total_variance_prior`` modules.
"""
import torch
from .core import RandomVariable


def _neighbor_diffs(volume):
    dims = list(range(volume.dim()))
    for d in dims:
        yield torch.diff(volume, dim=d)


class QuadraticSmoothnessRandomVariable(RandomVariable):
    """log p(x) = −(w/2) · ‖∇x‖²  (+ const): Gaussian smoothness energy."""

    normalized = False

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = float(weight)

    def log_prob(self, x):
        energy = sum(d.pow(2).sum() for d in _neighbor_diffs(x))
        return -0.5 * self.weight * energy


class TotalVariationRandomVariable(RandomVariable):
    """log p(x) = −w · TV(x)  (+ const): edge-preserving anisotropic TV energy."""

    normalized = False

    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = float(weight)

    def log_prob(self, x):
        energy = sum(d.abs().sum() for d in _neighbor_diffs(x))
        return -self.weight * energy


__all__ = [
    "QuadraticSmoothnessRandomVariable",
    "TotalVariationRandomVariable",
]
