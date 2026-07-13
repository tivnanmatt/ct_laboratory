import torch
from .linear_sde import LinearSDE
from ..linear_system import Scalar


class ScalarSDE(LinearSDE):
    def __init__(self, signal_scale, noise_variance, signal_scale_prime=None, noise_variance_prime=None):
        """
        This class implements a scalar stochastic differential equation (SDE).

        dx = f(t) x dt + g(t) dw

        Parameters:
            signal_scale: callable
                Function that returns a scalar representing the system response.
            noise_variance: callable
                Function that returns a scalar representing the covariance.
            signal_scale_prime: callable, optional
                Function that returns the time derivative of signal_scale. If not provided, it will be computed automatically.
            noise_variance_prime: callable, optional
                Function that returns the time derivative of noise_variance. If not provided, it will be computed automatically.
        """

        # handle the default case where signal_scale_prime is not provided, use autograd
        if signal_scale_prime is None:
            signal_scale_prime = lambda t: torch.autograd.grad(signal_scale(t), t, create_graph=True)[0]

        # handle the default case where noise_variance_prime is not provided, use autograd
        if noise_variance_prime is None:
            noise_variance_prime = lambda t: torch.autograd.grad(noise_variance(t), t, create_graph=True)[0]

        H = lambda t: Scalar(signal_scale(t))
        Sigma = lambda t: Scalar(noise_variance(t))

        H_prime = lambda t: Scalar(signal_scale_prime(t))
        Sigma_prime = lambda t: Scalar(noise_variance_prime(t))

        super(ScalarSDE, self).__init__(H, Sigma, H_prime, Sigma_prime) 