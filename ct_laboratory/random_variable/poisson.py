"""Poisson random variables, including the Beer–Lambert photon-counting model."""
import torch
from .core import RandomVariable, ConditionalRandomVariable


class PoissonRandomVariable(RandomVariable):
    """counts ~ Poisson(rate), independent per element."""

    def __init__(self, rate):
        super().__init__()
        self.register_buffer("rate", torch.as_tensor(rate))

    def _dist(self):
        return torch.distributions.Poisson(self.rate)

    def log_prob(self, x):
        return self._dist().log_prob(x).sum()

    def sample(self, sample_shape=torch.Size()):
        return self._dist().sample(sample_shape)


class ConditionalPoissonRandomVariable(ConditionalRandomVariable):
    """counts | x ~ Poisson(rate_fn(x))."""

    def __init__(self, rate_fn, eps=1e-12):
        super().__init__()
        self.rate_fn = rate_fn
        self.eps = eps

    def evaluate(self, given):
        rate = torch.clamp(self.rate_fn(given), min=self.eps)
        return PoissonRandomVariable(rate)


class BeerLambertPoissonRandomVariable(ConditionalPoissonRandomVariable):
    """Photon-counting CT measurement: counts | x ~ Poisson(I0 · exp(−op(x))).

    ``op`` is any line-integral operator closure (e.g. a projector's
    ``forward_project``); injecting it keeps this area free of tomography
    dependencies. ``x`` is the attenuation volume in the operator's units.
    """

    def __init__(self, op, I0=1e6, eps=1e-12):
        super().__init__(rate_fn=lambda x: I0 * torch.exp(-op(x)), eps=eps)
        self.op = op
        self.I0 = float(I0)


__all__ = [
    "PoissonRandomVariable",
    "ConditionalPoissonRandomVariable",
    "BeerLambertPoissonRandomVariable",
]
