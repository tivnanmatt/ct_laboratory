"""Random-variable interface (adapted from the gmi package for ct_laboratory).

A *random variable* is the same object a statistician calls a distribution —
we prefer the random-variable name. Two base classes:

* ``RandomVariable`` — unconditional: ``log_prob(x)``, ``sample()``,
  ``score(x)``. Priors are exactly unconditional random variables; no
  separate "prior" alias is needed.
* ``ConditionalRandomVariable`` — a family p(x | given): ``log_prob(x,
  given)``, ``sample(given)``, and ``evaluate(given) -> RandomVariable``.
  Likelihoods are exactly conditional random variables evaluated at the
  observed data; no separate "likelihood" alias is needed.

``log_pdf`` is provided as an alias of ``log_prob`` on both classes.
Interoperability with ``torch.distributions`` is first-class: see
``FromTorchDistribution`` / ``ConditionalFromTorchDistribution`` in
``from_torch_distribution.py``, and ``RandomVariable.from_torch`` below.

Note on normalization: improper energy-based variables (e.g. smoothness
priors) implement ``log_prob`` only up to an additive constant. That is fine
for MAP estimation and score-based methods; it is flagged by the
``normalized`` attribute.
"""
import torch
from torch import nn


class RandomVariable(nn.Module):
    """Unconditional random variable: exposes log_prob / sample / score."""

    #: False for energy-based variables whose log_prob omits the constant.
    normalized = True

    def log_prob(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def log_pdf(self, x):
        """Alias of :meth:`log_prob`."""
        return self.log_prob(x)

    def sample(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__} does not implement sampling")

    def score(self, x):
        """∇_x log p(x), computed by autograd (x need not require grad)."""
        x = x.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.log_prob(x), x, create_graph=True)
        return grad

    def forward(self, x):
        return self.log_prob(x)

    @staticmethod
    def from_torch(distribution):
        """Wrap a ``torch.distributions.Distribution`` as a RandomVariable."""
        from .from_torch_distribution import FromTorchDistribution
        return FromTorchDistribution(distribution)


class ConditionalRandomVariable(nn.Module):
    """Family of random variables p(x | given).

    Subclasses either override :meth:`evaluate` (returning the unconditional
    ``RandomVariable`` at a particular conditioning value) or override
    :meth:`log_prob` / :meth:`sample` directly.
    """

    normalized = True

    def evaluate(self, given):
        """Return the unconditional RandomVariable p(· | given)."""
        raise NotImplementedError

    def log_prob(self, x, given):
        return self.evaluate(given).log_prob(x)

    def log_pdf(self, x, given):
        """Alias of :meth:`log_prob`."""
        return self.log_prob(x, given)

    def sample(self, given, *args, **kwargs):
        return self.evaluate(given).sample(*args, **kwargs)

    def score(self, x, given):
        """∇_x log p(x | given) by autograd."""
        x = x.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(self.log_prob(x, given), x,
                                      create_graph=True)
        return grad

    def forward(self, x, given):
        return self.log_prob(x, given)
