
import torch
from torch import nn
import math

class DiagonalGaussianLogLikelihood(nn.Module):
    """
    Log-likelihood for y ~ N(mean, D{var})
    where D{var} is a diagonal covariance matrix.
    
    -log p(y | mean, var) = 0.5 * sum((y - mean)^2 / var) + 0.5 * sum(log(var)) + 0.5 * d * log(2*pi)
    """
    def __init__(self, y: torch.Tensor, var: torch.Tensor = None):
        super().__init__()
        self.y = y
        self.d = y.numel()
        
        if var is None:
            var = torch.ones_like(y)
        self.var = var
        
        # Precompute constant terms
        log_var_sum = torch.log(self.var).sum()
        self.const = -0.5 * (self.d * math.log(2 * math.pi) + log_var_sum).item()

    def forward(self, mean: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood given the mean.
        
        Args:
            mean: Predicted mean (e.g., Ax from projector)
            
        Returns:
            Log-likelihood value
        """
        term = -0.5 * ((self.y - mean) ** 2 / self.var).sum()
        return term + self.const
    

class LinearGaussianLogLikelihood(nn.Module):
    """
    Linear Gaussian likelihood model: y | x ~ N(Ax, D{var})
    
    where:
    - A is the forward projector
    - x is the image/volume
    - y are the measurements (projections)
    - D{var} is a diagonal covariance matrix (projection-dependent variance)
    """
    def __init__(self, projector, measurements=None, var=None, convert_to_attenuation=None):
        super().__init__()
        
        self.projector = projector
        
        if measurements is None:
            self.measurements = None
            self.gaussian_loss = None    
        else:
            self.set_measurements(measurements, var)
        
        if convert_to_attenuation is None:
            convert_to_attenuation = lambda x: x
        self.convert_to_attenuation = convert_to_attenuation

    def set_measurements(self, measurements: torch.Tensor, var: torch.Tensor = None):
        """
        Set the measurements and variance.
        
        Args:
            measurements: Observed projection data y
            var: Variance map (defaults to ones if None)
        """
        self.measurements = measurements
        self.gaussian_loss = DiagonalGaussianLogLikelihood(measurements, var)

    @torch.no_grad()
    def simulate_measurements(self, x, noise_std: float = 1.0):
        """
        Simulate noisy measurements from a ground truth image.
        
        Args:
            x: Ground truth image/volume
            noise_std: Standard deviation of Gaussian noise
            
        Returns:
            Noisy measurements y = Ax + noise
        """
        x_atten = self.convert_to_attenuation(x)
        sino_true = self.projector(x_atten)
        noise = torch.randn_like(sino_true) * noise_std
        measurements = sino_true + noise
        return measurements

    def forward(self, x):
        """
        Compute negative log-likelihood: -log p(y | x)
        
        Args:
            x: Image/volume
            
        Returns:
            Negative log-likelihood value
        """
        x_atten = self.convert_to_attenuation(x)
        sino_pred = self.projector(x_atten)
        return self.gaussian_loss(sino_pred)
    