import torch
from torch import nn
import math

class DiagonalGaussianLogPriorClosure(nn.Module):
    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.mu = mu.detach()
        self.std = std.detach()
        self.d = self.mu.numel()
        log_sigma2 = torch.log(self.std ** 2)
        self.const = 0.5 * (self.d * math.log(2 * math.pi) + log_sigma2.sum()).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        term = 0.5 * ((x - self.mu) ** 2 / (self.std ** 2)).sum()
        return term + self.const