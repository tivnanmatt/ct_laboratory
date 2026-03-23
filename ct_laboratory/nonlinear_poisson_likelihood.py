
import torch
from torch import nn

class PoissonLogLikelihood(nn.Module):
    def __init__(self, measured_counts: torch.Tensor):
        super().__init__()
        # self.register_buffer("y", y_counts)
        # self.register_buffer("c", torch.lgamma(y_counts + 1))
        self.measured_counts = measured_counts
        self.c = torch.lgamma(self.measured_counts + 1)

    def forward(self, expected_counts: torch.Tensor) -> torch.Tensor:
        return torch.sum(-expected_counts + self.measured_counts * torch.log(expected_counts) - self.c)
    
class NonlinearPoissonMeasurementLikelihood(nn.Module):
    def __init__(self, projector, I0=1e6, measured_counts=None, convert_to_attenuation=None):
        super().__init__()
        
        self.projector = projector
        self.I0 = I0
        
        if measured_counts is not None:
            self.set_measurements(measured_counts)
        else:
            self.measured_counts = None
            self.c = None
        
        if convert_to_attenuation is None:
            convert_to_attenuation = lambda x: x
        self.convert_to_attenuation = convert_to_attenuation

    def set_measurements(self, measured_counts: torch.Tensor):
        self.measured_counts = measured_counts
        self.c = torch.lgamma(self.measured_counts + 1)

    def compute_mean(self, x):
        """
        Compute the expected counts: I0 * exp(-A x)
        """
        x_atten = self.convert_to_attenuation(x)
        sino_pred = self.projector(x_atten)
        # Clamp sino_pred to avoid exp(70) or similar overflows if x_atten is very negative
        expected_counts = self.I0 * torch.exp(-torch.clamp(sino_pred, min=-20.0, max=100.0))
        return expected_counts

    @torch.no_grad()
    def sample(self, x):
        """
        Sample noisy measurements: Poisson(mean)
        """
        expected_counts = self.compute_mean(x)
        measured_counts = torch.poisson(expected_counts).float()
        return measured_counts

    def log_likelihood(self, x):
        """
        Compute the log-likelihood: sum(-mean + y*log(mean) - log(y!))
        Using log(I0 * exp(-s)) = log(I0) - s for numerical stability.
        """
        if self.measured_counts is None:
            raise ValueError("measured_counts must be set before computing log_likelihood")
            
        x_atten = self.convert_to_attenuation(x)
        sino_pred = self.projector(x_atten)
        
        # Avoid log(0) for I0
        log_I0 = torch.log(torch.tensor(self.I0, device=x.device, dtype=x.dtype) + 1e-12)
        
        # Stability: clamp sino_pred for the exponential term
        # If sino_pred is very large, exp(-sino_pred) is 0. 
        # If sino_pred is very negative, it explodes.
        sino_clamped = torch.clamp(sino_pred, min=-20.0, max=100.0)
        expected_counts = self.I0 * torch.exp(-sino_clamped)
        
        # log_lik = y * (log(I0) - s) - I0 * exp(-s) - log(y!)
        log_lik = self.measured_counts * (log_I0 - sino_pred) - expected_counts - self.c
        return torch.sum(log_lik)

    def forward(self, x):
        return self.log_likelihood(x)
    
class NonlinearPoissonLogLikelihood(NonlinearPoissonMeasurementLikelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import warnings
        warnings.warn("NonlinearPoissonLogLikelihood is deprecated, use NonlinearPoissonMeasurementLikelihood instead", DeprecationWarning)

    @torch.no_grad()
    def simulate_measurements(self, x_HU):
        return self.sample(x_HU)
    