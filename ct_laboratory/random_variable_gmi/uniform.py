import torch
from torch import nn

from .core import RandomVariable


class UniformRandomVariable(RandomVariable):
    def __init__(self, low=None, high=None):
        super(UniformRandomVariable, self).__init__()

        if low is None:
            low = torch.zeros(1)
        if high is None:
            high = torch.ones(1)

        if isinstance(low, (int, float)):
            low = torch.tensor(low).view(1)
        if isinstance(high, (int, float)):
            high = torch.tensor(high).view(1)

        assert isinstance(low, torch.Tensor)
        assert isinstance(high, torch.Tensor)
        assert low.size()[0] == 1
        assert high.size()[0] == 1
        assert low < high

        low = low.reshape(1)
        high = high.reshape(1)

        self.low = low
        self.high = high
        self.height = 1.0 / (high - low)

    def sample(self, batch_size):
        return torch.rand(batch_size, device=self.low.device) * (self.high - self.low) + self.low
    
    def log_prob(self, x):
        log_height = torch.log(self.height)
        x[torch.logical_or(x < self.low, x > self.high)] = -float('inf')
        x[torch.logical_and(x >= self.low, x <= self.high)] = log_height
        return x


# class GaussianRandomVariable(RandomVariable):
#     def __init__(self, mu, Sigma):
#         super(GaussianRandomVariable, self).__init__()

#         assert isinstance(mu, torch.Tensor)
#         assert isinstance(Sigma, LinearSystem)

#         self.mu = mu
#         self.Sigma = Sigma

#     def sample(self):
#         white_noise = torch.randn(self.mu.shape, device=self.mu.device)
#         sqrt_Sigma = self.Sigma.sqrt_LinearSystem()
#         correlated_noise =  sqrt_Sigma @ white_noise
#         return self.mu + correlated_noise
    
#     def mahalanobis_distance(self, x):
#         res = (x - self.mu)
#         weighted_res = self.Sigma.inv_LinearSystem() @ res
#         return torch.sum(res * weighted_res)
    
#     def log_prob(self, x):
#         d  = torch.prod(torch.tensor(self.mu.shape)).float()
#         constant_term = - d * torch.log(2 * torch.tensor([3.141592653589793]))
#         log_det = self.Sigma.logdet()
#         mahalanobis_distance = self.mahalanobis_distance(x)
#         return 0.5 * constant_term - 0.5 * log_det - 0.5 * mahalanobis_distance
    