"""Maximum a posteriori estimation on the random_variable / optimization areas.

The estimator is defined purely in probabilistic terms:

* likelihood — a ``ConditionalRandomVariable`` p(y | x); its ``log_prob`` is
  evaluated at the observed measurements. No separate likelihood class
  exists beyond conditional random variables.
* prior — an unconditional ``RandomVariable`` p(x) (possibly improper /
  energy-based). No separate prior class exists beyond unconditional random
  variables.
* posterior — p(x | y) ∝ p(y | x) · p(x); MAP maximizes its log, i.e.
  minimizes −(log p(y|x) + log p(x)) by preconditioned gradient descent.

Preconditioning takes a single :class:`ct_laboratory.optimization.Preconditioner`
(which is REQUIRED to implement both ``forward`` and ``inverse``) instead of
separate forward/inverse callables: optimization runs in the preconditioned
variable z with x = P(z), z0 = P^{-1}(x0).
"""
import time

import torch
from tqdm import tqdm

from ..optimization import Preconditioner, IdentityPreconditioner
from ..random_variable import RandomVariable, ConditionalRandomVariable


class MaximumAPosterioriEstimator:
    """MAP estimator: x* = argmax_x  log p(y_obs | x) + log p(x).

    Parameters
    ----------
    likelihood : ConditionalRandomVariable
        Measurement model p(y | x).
    prior : RandomVariable
        Prior p(x); improper energy-based variables are fine for MAP.
    measurements : tensor
        Observed data y_obs at which the likelihood is evaluated.
    x_init : tensor
        Initial estimate (defines shape/device of the optimization variable).
    preconditioner : Preconditioner, optional
        Change of variables x = P(z). Must be a
        ``ct_laboratory.optimization.Preconditioner`` (forward AND inverse);
        defaults to the identity.
    lr, warmup_iters : SGD learning rate and optional linear LR warmup.
    prior_weight : multiplier on the log prior (regularization strength).
    """

    def __init__(self, likelihood, prior, measurements, x_init,
                 preconditioner=None, lr=1.0, warmup_iters=None,
                 prior_weight=1.0):
        if not isinstance(likelihood, ConditionalRandomVariable):
            raise TypeError("likelihood must be a ConditionalRandomVariable "
                            f"(got {type(likelihood).__name__})")
        if not isinstance(prior, RandomVariable):
            raise TypeError("prior must be a RandomVariable "
                            f"(got {type(prior).__name__})")
        if preconditioner is None:
            preconditioner = IdentityPreconditioner()
        if not isinstance(preconditioner, Preconditioner):
            raise TypeError(
                "preconditioner must be a ct_laboratory.optimization."
                f"Preconditioner (got {type(preconditioner).__name__}); "
                "separate forward/inverse callables are no longer accepted")

        self.likelihood = likelihood
        self.prior = prior
        self.measurements = measurements
        self.preconditioner = preconditioner
        self.prior_weight = float(prior_weight)
        self._shape = x_init.shape

        with torch.no_grad():
            z0 = preconditioner.inverse(x_init.detach())
        self.z = z0.clone().detach().requires_grad_(True)
        self.optimizer = torch.optim.SGD([self.z], lr=lr)
        self.scheduler = None
        if warmup_iters is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda it: min(1.0, float(it) / warmup_iters))

    @property
    def x(self):
        """Current estimate in the original variable, x = P(z)."""
        with torch.no_grad():
            return self.preconditioner(self.z).reshape(self._shape)

    def negative_log_posterior(self, x):
        log_likelihood = self.likelihood.log_prob(self.measurements, x)
        log_prior = self.prior.log_prob(x)
        return -(log_likelihood + self.prior_weight * log_prior), \
            log_likelihood, log_prior

    def step(self):
        """One preconditioned gradient step; returns (loss, log_lik, log_prior)."""
        self.optimizer.zero_grad()
        x = self.preconditioner(self.z).reshape(self._shape)
        loss, log_likelihood, log_prior = self.negative_log_posterior(x)
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item(), log_likelihood.item(), log_prior.item()

    def estimate(self, num_iters=100, verbose=False, use_tqdm=True,
                 print_every=1):
        """Run ``num_iters`` steps and return the MAP estimate x*."""
        t0 = time.time()
        it_range = range(1, num_iters + 1)
        if use_tqdm:
            it_range = tqdm(it_range, desc="MAP estimation", leave=False)
        for it in it_range:
            loss, log_likelihood, log_prior = self.step()
            if use_tqdm:
                it_range.set_postfix({"loss": f"{loss:.3e}"})
            elif verbose and (it % print_every == 0 or it == 1):
                print(f"iter {it:04d}  -log_lik={-log_likelihood:.4e}  "
                      f"-log_prior={-log_prior:.4e}  loss={loss:.4e}  "
                      f"({time.time() - t0:.2f}s)")
                t0 = time.time()
        return self.x
