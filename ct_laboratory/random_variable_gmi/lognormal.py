import torch
from torch import nn
from .core import RandomVariable
from ..linear_system import LinearSystem, Scalar


class LogNormalRandomVariable(RandomVariable):
    def __init__(self, mu, Sigma):
        super(LogNormalRandomVariable, self).__init__()

        if isinstance(mu, int):
            mu = float(mu)

        if isinstance(Sigma, int):
            Sigma = float(Sigma)

        if isinstance(mu, float):
            mu = torch.tensor(mu)

        if isinstance(Sigma, float):
            Sigma = Scalar(Sigma)

        assert isinstance(mu, torch.Tensor)
        assert isinstance(Sigma, LinearSystem)

        self.mu = mu


        self.Sigma = Sigma

    def sample(self, batch_size):
        total_shape = [batch_size] + list(self.mu.shape)
        white_noise = torch.randn(total_shape, device=self.mu.device)
        sqrt_Sigma = self.Sigma.sqrt_LinearSystem()
        correlated_noise =  sqrt_Sigma @ white_noise
        return torch.exp(self.mu + correlated_noise)
    def log_prob(self, x):
        raise NotImplementedError("LogNormalRandomVariable does not support log_prob yet")