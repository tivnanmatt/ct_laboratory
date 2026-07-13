import torch
import pytest
from ct_laboratory.sde import LinearSDE
from ct_laboratory.linear_system import Scalar, DiagonalScalar

def test_linear_sde_instantiation():
    # Create simple H and Sigma functions
    H = lambda t: Scalar(2.0 * t + 1.0)
    Sigma = lambda t: Scalar(t**2 + 1.0)
    H_prime = lambda t: Scalar(2.0)
    Sigma_prime = lambda t: Scalar(2.0 * t)
    
    sde = LinearSDE(H=H, Sigma=Sigma, H_prime=H_prime, Sigma_prime=Sigma_prime)
    assert isinstance(sde, LinearSDE)

def test_linear_sde_mean_response():
    # Create simple H and Sigma functions
    H = lambda t: Scalar(2.0 * t + 1.0)
    Sigma = lambda t: Scalar(t**2 + 1.0)
    H_prime = lambda t: Scalar(2.0)
    Sigma_prime = lambda t: Scalar(2.0 * t)
    
    sde = LinearSDE(H=H, Sigma=Sigma, H_prime=H_prime, Sigma_prime=Sigma_prime)
    
    x0 = torch.randn(4)
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    expected = x0 * (2.0 * t + 1.0)
    assert torch.allclose(mean, expected, atol=1e-6)

def test_linear_sde_sample_shapes():
    # Create simple H and Sigma functions
    H = lambda t: Scalar(2.0 * t + 1.0)
    Sigma = lambda t: Scalar(t**2 + 1.0)
    H_prime = lambda t: Scalar(2.0)
    Sigma_prime = lambda t: Scalar(2.0 * t)
    
    sde = LinearSDE(H=H, Sigma=Sigma, H_prime=H_prime, Sigma_prime=Sigma_prime)
    
    x0 = torch.randn(4)
    t = torch.tensor(0.5)
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape
    
    noise = torch.randn_like(x0)
    xt2 = sde.sample_x_t_given_x_0_and_noise(x0, noise, t)
    assert xt2.shape == x0.shape

def test_linear_sde_diagonal():
    # Test with diagonal operators
    H = lambda t: DiagonalScalar(torch.ones(4) * (2.0 * t + 1.0))
    Sigma = lambda t: DiagonalScalar(torch.ones(4) * (t**2 + 1.0))
    H_prime = lambda t: DiagonalScalar(torch.ones(4) * 2.0)
    Sigma_prime = lambda t: DiagonalScalar(torch.ones(4) * (2.0 * t))
    
    sde = LinearSDE(H=H, Sigma=Sigma, H_prime=H_prime, Sigma_prime=Sigma_prime)
    
    x0 = torch.randn(4)
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    expected = x0 * (2.0 * t + 1.0)
    assert torch.allclose(mean, expected, atol=1e-6)
    
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape

def test_linear_sde_reverse_score_estimator():
    # Create simple H and Sigma functions
    H = lambda t: Scalar(2.0 * t + 1.0)
    Sigma = lambda t: Scalar(t**2 + 1.0)
    H_prime = lambda t: Scalar(2.0)
    Sigma_prime = lambda t: Scalar(2.0 * t)
    
    sde = LinearSDE(H=H, Sigma=Sigma, H_prime=H_prime, Sigma_prime=Sigma_prime)
    
    def dummy_score(x, t):
        return torch.zeros_like(x)
    
    rev = sde.reverse_SDE_given_score_estimator(dummy_score)
    x = torch.randn(4)
    t = torch.tensor(0.5)
    f, G = rev(x, t)
    assert isinstance(f, torch.Tensor)
    assert f.shape == x.shape

def test_linear_sde_reverse_mean_estimator():
    # Create simple H and Sigma functions
    H = lambda t: Scalar(2.0 * t + 1.0)
    Sigma = lambda t: Scalar(t**2 + 1.0)
    H_prime = lambda t: Scalar(2.0)
    Sigma_prime = lambda t: Scalar(2.0 * t)
    
    sde = LinearSDE(H=H, Sigma=Sigma, H_prime=H_prime, Sigma_prime=Sigma_prime)
    
    def dummy_mean(x, t):
        return x
    
    rev = sde.reverse_SDE_given_mean_estimator(dummy_mean)
    x = torch.randn(4)
    t = torch.tensor(0.5)
    f, G = rev(x, t)
    assert isinstance(f, torch.Tensor)
    assert f.shape == x.shape 