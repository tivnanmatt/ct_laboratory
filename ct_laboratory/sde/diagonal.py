# random_tensor_laboratory/diffusion/sde.py

import torch
import torch
from .linear_sde import LinearSDE
from ..linear_system import DiagonalScalar

class DiagonalSDE(LinearSDE):
    def __init__(self, signal_scale, noise_variance, signal_scale_prime=None, noise_variance_prime=None):
        """
        This class implements a diagonal stochastic differential equation (SDE).

        Parameters:
            signal_scale: callable
                Function that returns a diagonal vector representing the system response.
            noise_variance: callable
                Function that returns a diagonal vector representing the covariance.
            signal_scale_prime: callable, optional
                Function that returns the time derivative of signal_scale. If not provided, it will be computed automatically.
            noise_variance_prime: callable, optional
                Function that returns the time derivative of noise_variance. If not provided, it will be computed automatically.
        """
        assert isinstance(signal_scale(0.0), (torch.Tensor)), "signal_scale(t) must return a diagonal vector."
        assert isinstance(noise_variance(0.0), (torch.Tensor)), "noise_variance(t) must return a diagonal vector."

        H = lambda t: DiagonalScalar(signal_scale(t))
        Sigma = lambda t: DiagonalScalar(noise_variance(t))

        if signal_scale_prime is None:
            signal_scale_prime = lambda t: torch.autograd.grad(signal_scale(t), t, create_graph=True)[0]
        if noise_variance_prime is None:
            noise_variance_prime = lambda t: torch.autograd.grad(noise_variance(t), t, create_graph=True)[0]

        H_prime = lambda t: DiagonalScalar(signal_scale_prime(t))
        Sigma_prime = lambda t: DiagonalScalar(noise_variance_prime(t))

        super(DiagonalSDE, self).__init__(H, Sigma, H_prime, Sigma_prime)
