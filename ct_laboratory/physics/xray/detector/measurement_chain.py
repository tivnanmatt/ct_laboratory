"""Raw measurement chain and likelihoods.

    raw_i = o_j + gamma_j N_i + eps_i,      N_i ~ Poisson(lambda_i),   eps_i ~ N(0, sigma_j^2)
    z_i   = (raw_i - o_j) / gamma_j + kappa_j  ~  Poisson(lambda_i + kappa_j),   kappa_j = sigma_j^2 / gamma_j^2   (shifted Poisson)

o_j offset (ADU), gamma_j gain (ADU per photon-equivalent; absorbs the energy weighting of S_2), sigma_j read noise (ADU).

Two likelihoods are provided:
  * ShiftedPoissonChain.nll        exact shifted-Poisson deviance on photon-equivalent counts (reconstruction)
  * AirNormalizedGaussian          Gaussian on t = (z - kappa) / I_0 with the Poisson + read-noise variance plus the
                                   nuisance terms used for calibration fits:
        var(t) = (t I_0 + kappa) / I_0^2  +  floor^2  +  (jitter * |dt/dcol|)^2  +  (dm/dL)^2 sigma_L^2
    floor   multiplicative model error as a fraction of air (prior 0.3 % x/: e)
    jitter  sub-pixel misregistration of shadow edges in pixels (prior 0.1 px x/: e)
    sigma_L errors-in-variables term: uncertainty of the object path length (supplied by the object model)
"""
import math
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters


class ShiftedPoissonChain(torch.nn.Module):
    def __init__(self, gain, offset, read_noise):
        super().__init__()
        self.register_buffer('gain', torch.as_tensor(gain, dtype=torch.float32))
        self.register_buffer('offset', torch.as_tensor(offset, dtype=torch.float32))
        self.register_buffer('read_noise', torch.as_tensor(read_noise, dtype=torch.float32))

    @property
    def kappa(self):
        return (self.read_noise / self.gain) ** 2

    def shifted_counts(self, raw):
        """z = (raw - o) / gamma + kappa"""
        return (raw - self.offset) / self.gain + self.kappa

    def nll(self, z, lam, mask=None):
        """sum over valid i of (lambda + kappa) - z log(lambda + kappa)   (z already shifted)"""
        m = lam + self.kappa
        v = m - z * torch.log(m.clamp(min=1e-12))
        return (v if mask is None else torch.where(mask, v, torch.zeros_like(v))).sum()

    def sample(self, lam, generator=None):
        """raw ADU sample of the chain for mean counts lam"""
        n = torch.poisson(lam.clamp(min=0), generator=generator)
        return self.offset + self.gain * n + self.read_noise * torch.randn(n.shape, generator=generator, device=n.device)


class AirNormalizedGaussian(torch.nn.Module):
    def __init__(self, floor=0.003, floor_sd=1.0, jitter_px=0.1, jitter_sd=1.0):
        super().__init__()
        p = PriorParameters()
        p.add('floor', floor, floor_sd, 'exp', label='noise floor (% of air)', display=lambda v: 100 * v)
        p.add('jitter_px', jitter_px, jitter_sd, 'exp', label='edge jitter (px)')
        self.params = p

    def variance(self, t, I0, kappa, grad_col=None, dm_dL=None, sigma_L=None):
        v = (t.clamp(min=1e-4) * I0 + kappa) / I0 ** 2 + self.params['floor'] ** 2
        if grad_col is not None:
            v = v + (self.params['jitter_px'] * grad_col) ** 2
        if dm_dL is not None and sigma_L is not None:
            v = v + (dm_dL * sigma_L) ** 2
        return v

    @staticmethod
    def nll(t, m, v, mask):
        z = torch.zeros_like(t)
        return 0.5 * torch.where(mask, (t - m) ** 2 / v + torch.log(v), z).double().sum()

    @staticmethod
    def profiled_scale(t, m0, v, mask):
        """closed-form per-firing output factor s = argmin sum (t - s m0)^2 / v  (treated as constant in the gradient)"""
        z = torch.zeros_like(t)
        return (torch.where(mask, t * m0 / v, z).sum() / torch.where(mask, m0 * m0 / v, z).sum().clamp(min=1e-9)).detach()
