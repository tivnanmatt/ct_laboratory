"""General loss functions for gradient-based optimization.

These are probability-agnostic: a loss is any differentiable scalar-valued
function of the optimization variable. Probabilistic objectives (negative log
posteriors, etc.) are built ON TOP of these in ``bayesian_estimation``; this
module must not import ``random_variable`` or any projector code.
"""
import torch
from torch import nn


class LossFunction(nn.Module):
    """Base class: a differentiable scalar loss ``forward(x) -> ()`` tensor."""

    def forward(self, x):  # pragma: no cover - abstract
        raise NotImplementedError


class WeightedSumLoss(LossFunction):
    """weighted sum of loss terms:  L(x) = sum_i w_i * L_i(x).

    Terms are arbitrary callables returning scalars, so both ``LossFunction``
    instances and plain closures compose.
    """

    def __init__(self, terms, weights=None):
        super().__init__()
        self.terms = nn.ModuleList(
            [t for t in terms if isinstance(t, nn.Module)])
        self._all_terms = list(terms)
        self.weights = [1.0] * len(terms) if weights is None else list(weights)
        if len(self.weights) != len(self._all_terms):
            raise ValueError("weights must match number of terms")

    def forward(self, x):
        total = None
        for w, term in zip(self.weights, self._all_terms):
            val = w * term(x)
            total = val if total is None else total + val
        return total


class QuadraticLoss(LossFunction):
    """L(x) = 0.5 * ||op(x) - target||^2 for an arbitrary callable ``op``.

    ``op`` defaults to the identity; pass any linear operator closure (a
    projector's ``forward_project``, a blur, ...) to build data-fidelity terms
    without this module depending on those operators.
    """

    def __init__(self, target, op=None):
        super().__init__()
        self.register_buffer("target", torch.as_tensor(target))
        self.op = op if op is not None else (lambda x: x)

    def forward(self, x):
        residual = self.op(x).reshape(-1) - self.target.reshape(-1)
        return 0.5 * torch.dot(residual, residual)
