"""Abstract preconditioner interface.

A preconditioner is a symmetric positive-definite change of variables
``x = P(z)`` used to accelerate gradient-based optimization. Every concrete
preconditioner MUST implement both directions:

* ``forward(x)`` — apply P
* ``inverse(x)`` — apply P^-1

Subclassing is checked at class-definition time: a subclass that fails to
override either method raises ``TypeError`` immediately, so a missing
``inverse`` can never surface later as a silent runtime bug. Optimizers in
other areas (e.g. ``bayesian_estimation``) accept a single ``Preconditioner``
instance instead of separate forward/inverse callables.
"""
import torch
from torch import nn


class Preconditioner(nn.Module):
    """Base class for preconditioners: subclasses must implement forward AND inverse."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for required in ("forward", "inverse"):
            impl = cls.__dict__.get(required)
            inherited = getattr(cls, required, None)
            if impl is None and (inherited is None
                                 or inherited in (Preconditioner.forward,
                                                  Preconditioner.inverse)):
                raise TypeError(
                    f"Preconditioner subclass {cls.__name__!r} must implement "
                    f"{required!r} (both forward and inverse are required)")

    def forward(self, x):  # pragma: no cover - abstract
        raise NotImplementedError("Preconditioner.forward must be implemented")

    def inverse(self, x):  # pragma: no cover - abstract
        raise NotImplementedError("Preconditioner.inverse must be implemented")


class IdentityPreconditioner(Preconditioner):
    """P = I. Useful default; makes preconditioned code paths uniform."""

    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class DiagonalPreconditioner(Preconditioner):
    """P = diag(d) with elementwise positive scale ``d`` (tensor or scalar)."""

    def __init__(self, scale):
        super().__init__()
        scale = torch.as_tensor(scale)
        if torch.any(scale <= 0):
            raise ValueError("DiagonalPreconditioner scale must be positive")
        self.register_buffer("scale", scale)

    def forward(self, x):
        return x * self.scale

    def inverse(self, x):
        return x / self.scale
