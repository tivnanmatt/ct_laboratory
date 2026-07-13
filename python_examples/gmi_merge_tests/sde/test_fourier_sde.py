import torch
import pytest
from ct_laboratory.sde import FourierSDE

def test_fourier_sde_instantiation():
    # Create simple transfer function and noise power spectrum
    transfer_function = lambda t: torch.ones(4) * (1.0 + 0.5 * t)
    noise_power_spectrum = lambda t: torch.ones(4) * (t + 0.1)
    transfer_function_prime = lambda t: torch.ones(4) * 0.5
    noise_power_spectrum_prime = lambda t: torch.ones(4) * 1.0
    
    sde = FourierSDE(transfer_function=transfer_function, 
                     noise_power_spectrum=noise_power_spectrum,
                     dim=0,  # Use dimension 0 for 1D tensor
                     transfer_function_prime=transfer_function_prime,
                     noise_power_spectrum_prime=noise_power_spectrum_prime)
    assert isinstance(sde, FourierSDE)

def test_fourier_sde_mean_response():
    # Create simple transfer function and noise power spectrum
    transfer_function = lambda t: torch.ones(4) * (1.0 + 0.5 * t)
    noise_power_spectrum = lambda t: torch.ones(4) * (t + 0.1)
    transfer_function_prime = lambda t: torch.ones(4) * 0.5
    noise_power_spectrum_prime = lambda t: torch.ones(4) * 1.0
    
    sde = FourierSDE(transfer_function=transfer_function, 
                     noise_power_spectrum=noise_power_spectrum,
                     dim=0,  # Use dimension 0 for 1D tensor
                     transfer_function_prime=transfer_function_prime,
                     noise_power_spectrum_prime=noise_power_spectrum_prime)
    
    x0 = torch.randn(4)
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    assert mean.shape == x0.shape

def test_fourier_sde_sample_shapes():
    # Create simple transfer function and noise power spectrum
    transfer_function = lambda t: torch.ones(4) * (1.0 + 0.5 * t)
    noise_power_spectrum = lambda t: torch.ones(4) * (t + 0.1)
    transfer_function_prime = lambda t: torch.ones(4) * 0.5
    noise_power_spectrum_prime = lambda t: torch.ones(4) * 1.0
    
    sde = FourierSDE(transfer_function=transfer_function, 
                     noise_power_spectrum=noise_power_spectrum,
                     dim=0,  # Use dimension 0 for 1D tensor
                     transfer_function_prime=transfer_function_prime,
                     noise_power_spectrum_prime=noise_power_spectrum_prime)
    
    x0 = torch.randn(4)
    t = torch.tensor(0.5)
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape
    
    noise = torch.randn_like(x0)
    xt2 = sde.sample_x_t_given_x_0_and_noise(x0, noise, t)
    assert xt2.shape == x0.shape

def test_fourier_sde_2d():
    # Test 2D Fourier SDE
    transfer_function = lambda t: torch.ones(4, 4) * (1.0 + 0.5 * t)
    noise_power_spectrum = lambda t: torch.ones(4, 4) * (t + 0.1)
    transfer_function_prime = lambda t: torch.ones(4, 4) * 0.5
    noise_power_spectrum_prime = lambda t: torch.ones(4, 4) * 1.0
    
    sde = FourierSDE(transfer_function=transfer_function, 
                     noise_power_spectrum=noise_power_spectrum,
                     dim=(0, 1),  # Use dimensions 0 and 1 for 2D tensor
                     transfer_function_prime=transfer_function_prime,
                     noise_power_spectrum_prime=noise_power_spectrum_prime)
    
    x0 = torch.randn(4, 4)
    t = torch.tensor(0.5)
    mean = sde.mean_response_x_t_given_x_0(x0, t)
    assert mean.shape == x0.shape
    
    xt = sde.sample_x_t_given_x_0(x0, t)
    assert xt.shape == x0.shape 