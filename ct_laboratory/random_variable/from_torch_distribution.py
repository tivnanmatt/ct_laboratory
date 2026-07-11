"""Bridges between torch.distributions and the RandomVariable interface.

Random variables and torch Distributions are essentially the same thing; we
prefer the random-variable name. These wrappers make any
``torch.distributions.Distribution`` usable wherever a ``RandomVariable`` is
expected (and, via a distribution-factory, wherever a
``ConditionalRandomVariable`` is expected).
"""
import torch
from .core import RandomVariable, ConditionalRandomVariable


class FromTorchDistribution(RandomVariable):
    """Wrap a torch.distributions.Distribution.

    ``log_prob`` sums the per-element log densities to one scalar (the joint
    log probability of the whole tensor under independent components), which
    is the convention every estimator in ct_laboratory expects.
    """

    def __init__(self, distribution):
        super().__init__()
        if not isinstance(distribution, torch.distributions.Distribution):
            raise TypeError("expected a torch.distributions.Distribution, "
                            f"got {type(distribution).__name__}")
        self.distribution = distribution

    def log_prob(self, x):
        return self.distribution.log_prob(x).sum()

    def sample(self, sample_shape=torch.Size()):
        return self.distribution.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.distribution.rsample(sample_shape)


class ConditionalFromTorchDistribution(ConditionalRandomVariable):
    """Conditional variable from a factory ``given -> torch Distribution``.

    Example — a Poisson measurement model:

        rv = ConditionalFromTorchDistribution(
            lambda x: torch.distributions.Poisson(rate_fn(x)))
        rv.log_prob(counts, x)
    """

    def __init__(self, distribution_fn):
        super().__init__()
        self.distribution_fn = distribution_fn

    def evaluate(self, given):
        return FromTorchDistribution(self.distribution_fn(given))
