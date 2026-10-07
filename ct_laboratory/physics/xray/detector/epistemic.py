"""Epistemic (model-error) term: per-pixel multiplicative gain error, shared by all scans of the same pixel.

    m_{k,j} -> m_{k,j} (1 + eps_j),     eps_j ~ N(0, sigma_e^2)

eps_j absorbs everything that is fixed per detector pixel and not in the physics model (flat-field / gain-map errors, pixel
non-linearity at this signal level, local model misfit).  sigma_e is the "epistemic dial": 0 -> off, large -> the data of each pixel are
fitted almost perfectly.  For a Gaussian likelihood the optimum is closed form per pixel (profiled, differentiable):

    eps_j = sum_k m (t - m) / v  /  ( sum_k m^2 / v + 1 / sigma_e^2 )

CAUTION: when every scan sees the same path through the object at pixel j (e.g. a z-uniform phantom stepped along z), eps_j is fully
confounded with the object model at that pixel; interpret corrected-data tests accordingly.
"""
import torch


class PixelGainError(torch.nn.Module):
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = float(sigma)

    def profile(self, t, m, v, mask):
        """t, m, v, mask [n_scans, ...] -> eps [...] (zeros when sigma = 0)"""
        if self.sigma <= 0:
            return torch.zeros_like(m[0])
        z = torch.zeros_like(m)
        num = torch.where(mask, m * (t - m) / v, z).sum(0)
        den = torch.where(mask, m * m / v, z).sum(0) + 1.0 / self.sigma ** 2
        return num / den

    def neg_log_prior(self, eps):
        return 0.0 if self.sigma <= 0 else 0.5 * ((eps / self.sigma) ** 2).double().sum()
