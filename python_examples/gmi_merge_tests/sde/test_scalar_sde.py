import torch
import pytest
from ct_laboratory.sde import ScalarSDE

def test_scalar_sde_instantiation():
    sde = ScalarSDE(signal_scale=lambda t: 2.0 * t + 1.0, noise_variance=lambda t: t**2 + 1.0,
                    signal_scale_prime=lambda t: 2.0, noise_variance_prime=lambda t: 2.0 * t)
    assert isinstance(sde, ScalarSDE)

def test_scalar_sde_mean_response(sample_tensor_4d):
    sde = ScalarSDE(signal_scale=lambda t: 2.0 * t + 1.0, noise_variance=lambda t: t**2 + 1.0,
                    signal_scale_prime=lambda t: 2.0, noise_variance_prime=lambda t: 2.0 * t)
    x0 = sample_tensor_4d
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    expected = x0 * (2.0 * t + 1.0)
    assert torch.allclose(mean, expected, atol=1e-6)

def test_scalar_sde_sample_shapes(sample_tensor_4d):
    sde = ScalarSDE(signal_scale=lambda t: 2.0 * t + 1.0, noise_variance=lambda t: t**2 + 1.0,
                    signal_scale_prime=lambda t: 2.0, noise_variance_prime=lambda t: 2.0 * t)
    x0 = sample_tensor_4d
    t = torch.tensor(0.5)
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape
    noise = torch.randn_like(x0)
    xt2 = sde.sample_x_t_given_x_0_and_noise(x0, noise, t)
    assert xt2.shape == x0.shape

def test_scalar_sde_variance_increases(sample_tensor_4d):
    sde = ScalarSDE(signal_scale=lambda t: 1.0, noise_variance=lambda t: t + 1.0,
                    signal_scale_prime=lambda t: 0.0, noise_variance_prime=lambda t: 1.0)
    x0 = sample_tensor_4d
    timesteps = torch.linspace(0, 1, 10)
    xT = sde.sample(x0, timesteps)
    var0 = x0.var().item()
    varT = xT.var().item()
    assert varT > var0 