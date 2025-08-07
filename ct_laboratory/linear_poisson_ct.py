
import torch
from torch import nn


class PoissonNegativeLogLikelihood(nn.Module):
    def __init__(self, y_counts: torch.Tensor):
        super().__init__()
        # self.register_buffer("y", y_counts)
        # self.register_buffer("c", torch.lgamma(y_counts + 1))
        self.y = y_counts
        self.c = torch.lgamma(y_counts + 1)

    def forward(self, rate_pred: torch.Tensor) -> torch.Tensor:
        return torch.sum(rate_pred - self.y * torch.log(rate_pred) + self.c)
    


class LinearPoissonCTLossClosure(nn.Module):
    def __init__(self, projector, I0=1e6, measured_counts=None, convert_to_attenuation=None):
        super().__init__()
        self.projector = projector
        self.I0 = I0
        self.measured_counts = None
        self.poisson_loss = None
        if measured_counts is not None:
            self.set_measurements(measured_counts)
        if convert_to_attenuation is not None:
            self.convert_to_attenuation = convert_to_attenuation
        else:
            self.convert_to_attenuation = lambda x: x

    def set_measurements(self, measured_counts: torch.Tensor):
        self.measured_counts = measured_counts
        self.poisson_loss = PoissonNegativeLogLikelihood(measured_counts)

    @torch.no_grad()
    def simulate_measurements(self, x_HU):
        x_atten = self.convert_to_attenuation(x_HU)
        sino_true = self.projector(x_atten)
        expected_counts = self.I0 * torch.exp(-sino_true)
        counts = torch.poisson(expected_counts).float()
        counts[counts == 0] = 1
        return counts

    def forward(self, x_HU):
        # x_atten = (x_HU + 1000) * (self.atten_coeff / 1000)
        x_atten = self.convert_to_attenuation(x_HU)
        sino_pred = self.projector(x_atten)
        counts_pred = self.I0 * torch.exp(-sino_pred)
        return self.poisson_loss(counts_pred)
    