"""Gaussian random variables (diagonal covariance), built on torch.distributions."""
import torch
from .core import RandomVariable, ConditionalRandomVariable
from .from_torch_distribution import FromTorchDistribution


class DiagonalGaussianRandomVariable(RandomVariable):
    """x ~ N(mu, diag(var)). ``var`` may be a scalar or a tensor broadcastable
    to ``mu``."""

    def __init__(self, mu, var=1.0):
        super().__init__()
        mu = torch.as_tensor(mu)
        var = torch.as_tensor(var, dtype=mu.dtype)
        self.register_buffer("mu", mu)
        self.register_buffer("var", torch.broadcast_to(var, mu.shape).clone())

    def _dist(self):
        return torch.distributions.Normal(self.mu, self.var.sqrt())

    def log_prob(self, x):
        return self._dist().log_prob(x).sum()

    def sample(self, sample_shape=torch.Size()):
        return self._dist().sample(sample_shape)

    def score(self, x):
        return (self.mu - x) / self.var


class ConditionalGaussianRandomVariable(ConditionalRandomVariable):
    """x | given ~ N(mean_fn(given), diag(var_fn(given))).

    ``var_fn`` may be omitted (unit variance), a constant, or a callable of
    the conditioning value.
    """

    def __init__(self, mean_fn, var_fn=1.0):
        super().__init__()
        self.mean_fn = mean_fn
        self.var_fn = var_fn if callable(var_fn) else (lambda given, v=var_fn: v)

    def evaluate(self, given):
        return DiagonalGaussianRandomVariable(self.mean_fn(given),
                                              self.var_fn(given))


class LinearGaussianRandomVariable(ConditionalGaussianRandomVariable):
    """y | x ~ N(op(x), diag(var)) for an arbitrary linear operator closure.

    ``op`` is any callable (e.g. a projector's ``forward_project``); this
    module has no dependency on tomography — the operator is injected.
    """

    def __init__(self, op, var=1.0):
        super().__init__(mean_fn=lambda x: op(x), var_fn=var)
        self.op = op


class AdditiveWhiteGaussianNoise(ConditionalGaussianRandomVariable):
    """y | x ~ N(x, sigma^2 I) — pure measurement-noise channel."""

    def __init__(self, noise_standard_deviation):
        super().__init__(mean_fn=lambda x: x,
                         var_fn=float(noise_standard_deviation) ** 2)
        self.noise_standard_deviation = float(noise_standard_deviation)


__all__ = [
    "DiagonalGaussianRandomVariable",
    "ConditionalGaussianRandomVariable",
    "LinearGaussianRandomVariable",
    "AdditiveWhiteGaussianNoise",
    "FromTorchDistribution",
]
