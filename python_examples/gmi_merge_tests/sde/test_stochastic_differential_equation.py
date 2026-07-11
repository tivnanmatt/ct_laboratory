import torch
import pytest
from ct_laboratory.sde import StochasticDifferentialEquation
from ct_laboratory.linear_system import Scalar

def test_sde_instantiation():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    assert isinstance(sde, StochasticDifferentialEquation)

def test_sde_forward():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    
    x = torch.randn(4)
    t = torch.tensor(0.5)
    f_val, G_op = sde(x, t)
    
    assert isinstance(f_val, torch.Tensor)
    assert f_val.shape == x.shape
    assert isinstance(G_op, Scalar)

def test_sde_sample_shapes():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    
    x0 = torch.randn(4)
    timesteps = torch.linspace(0, 1, 10)
    xT = sde.sample(x0, timesteps)
    
    assert xT.shape == x0.shape

def test_sde_sample_return_all():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    
    x0 = torch.randn(4)
    timesteps = torch.linspace(0, 1, 5)
    x_all = sde.sample(x0, timesteps, return_all=True)
    
    assert isinstance(x_all, list)
    assert len(x_all) == len(timesteps)
    for x in x_all:
        assert x.shape == x0.shape

def test_sde_euler_sampler():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    
    x0 = torch.randn(4)
    timesteps = torch.linspace(0, 1, 10)
    xT = sde.sample(x0, timesteps, sampler='euler')
    
    assert xT.shape == x0.shape

def test_sde_heun_sampler():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    
    x0 = torch.randn(4)
    timesteps = torch.linspace(0, 1, 10)
    xT = sde.sample(x0, timesteps, sampler='heun')
    
    assert xT.shape == x0.shape

def test_sde_invalid_sampler():
    # Create simple drift and diffusion functions
    def f(x, t):
        return -x  # Simple mean-reverting drift
    
    def G(x, t):
        return Scalar(1.0)  # Constant diffusion
    
    sde = StochasticDifferentialEquation(f=f, G=G)
    
    x0 = torch.randn(4)
    timesteps = torch.linspace(0, 1, 10)
    
    with pytest.raises(ValueError, match="The sampler should be one of"):
        sde.sample(x0, timesteps, sampler='invalid') 