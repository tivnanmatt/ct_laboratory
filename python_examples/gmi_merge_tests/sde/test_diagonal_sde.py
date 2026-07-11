import torch
import pytest
from ct_laboratory.sde import DiagonalSDE

def test_diagonal_sde_instantiation():
    signal_scale = lambda t: torch.ones(4) * (2.0 * t + 1.0)
    noise_variance = lambda t: torch.ones(4) * (t**2 + 1.0)
    signal_scale_prime = lambda t: torch.ones(4) * 2.0
    noise_variance_prime = lambda t: torch.ones(4) * (2.0 * t)
    sde = DiagonalSDE(signal_scale=signal_scale, noise_variance=noise_variance,
                      signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime)
    assert isinstance(sde, DiagonalSDE)

def test_diagonal_sde_mean_response():
    signal_scale = lambda t: torch.ones(4) * (2.0 * t + 1.0)
    noise_variance = lambda t: torch.ones(4) * (t**2 + 1.0)
    signal_scale_prime = lambda t: torch.ones(4) * 2.0
    noise_variance_prime = lambda t: torch.ones(4) * (2.0 * t)
    sde = DiagonalSDE(signal_scale=signal_scale, noise_variance=noise_variance,
                      signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime)
    x0 = torch.arange(4.0)
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    expected = x0 * (2.0 * t + 1.0)
    assert torch.allclose(mean, expected, atol=1e-6)

def test_diagonal_sde_sample_shapes():
    signal_scale = lambda t: torch.ones(4) * (2.0 * t + 1.0)
    noise_variance = lambda t: torch.ones(4) * (t**2 + 1.0)
    signal_scale_prime = lambda t: torch.ones(4) * 2.0
    noise_variance_prime = lambda t: torch.ones(4) * (2.0 * t)
    sde = DiagonalSDE(signal_scale=signal_scale, noise_variance=noise_variance,
                      signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime)
    x0 = torch.arange(4.0)
    t = torch.tensor(0.5)
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape
    noise = torch.randn_like(x0)
    xt2 = sde.sample_x_t_given_x_0_and_noise(x0, noise, t)
    assert xt2.shape == x0.shape

def test_diagonal_sde_variance_increases():
    signal_scale = lambda t: torch.ones(4)
    noise_variance = lambda t: torch.ones(4) * (t + 1.0)
    signal_scale_prime = lambda t: torch.zeros(4)
    noise_variance_prime = lambda t: torch.ones(4)
    sde = DiagonalSDE(signal_scale=signal_scale, noise_variance=noise_variance,
                      signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime)
    x0 = torch.arange(4.0)
    timesteps = torch.linspace(0, 100, 10)
    xT = sde.sample(x0, timesteps)
    var0 = x0.var().item()
    varT = xT.var().item()
    assert varT > var0 
