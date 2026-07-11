# random_tensor_laboratory/diffusion/sde.py

import torch
import torch
from .linear_sde import LinearSDE
from ..linear_system import FourierFilter

class FourierSDE(LinearSDE):
    def __init__(self, transfer_function, noise_power_spectrum, dim, transfer_function_prime=None, noise_power_spectrum_prime=None):
        """
        This class implements a Fourier stochastic differential equation (SDE).

        Parameters:
            transfer_function: callable
                Function that returns a Fourier filter representing the system response.
            noise_power_spectrum: callable
                Function that returns a Fourier filter representing the covariance.
            dim: int
                The number of dimensions for the Fourier transform.
            transfer_function_prime: callable, optional
                Function that returns the time derivative of transfer_function. If not provided, it will be computed automatically.
            noise_power_spectrum_prime: callable, optional
                Function that returns the time derivative of noise_power_spectrum. If not provided, it will be computed automatically.
        """

        assert isinstance(transfer_function(0.0), (torch.Tensor)), "transfer_function(t) must return a Fourier filter."
        assert isinstance(noise_power_spectrum(0.0), (torch.Tensor)), "noise_power_spectrum(t) must return a Fourier filter."

        H = lambda t: FourierFilter(transfer_function(t), dim)
        Sigma = lambda t: FourierFilter(noise_power_spectrum(t), dim)

        if transfer_function_prime is None:
            transfer_function_prime = lambda t: torch.autograd.grad(transfer_function(t), t, create_graph=True)[0]
        if noise_power_spectrum_prime is None:
            noise_power_spectrum_prime = lambda t: torch.autograd.grad(noise_power_spectrum(t), t, create_graph=True)[0]

        H_prime = lambda t: FourierFilter(transfer_function_prime(t), dim)
        Sigma_prime = lambda t: FourierFilter(noise_power_spectrum_prime(t), dim)

        super(FourierSDE, self).__init__(H, Sigma, H_prime, Sigma_prime)


    
        



# some ideas for the future

# class OrnsteinUhlenbeckProcess(nn.Module):
# class GeometricBrownianMotion(nn.Module):
# class VasicekModel(nn.Module):
# class CIRModel(nn.Module):
# class HestonModel(nn.Module):
# class MertonJumpDiffusionModel(nn.Module):
# class KouJumpDiffusionModel(nn.Module):
# class VarianceGammaProcess(nn.Module):
# class NormalInverseGaussianProcess(nn.Module):
# class MeixnerProcess(nn.Module):
# class GeneralizedHyperbolicProcess(nn.Module):
# class NormalInverseGaussianProcess(nn.Module):


    
# class SongVarianceExploding(nn.Module):
#     def __init__(self, sigma_1=80.0):
#         super(SongVarianceExploding, self).__init__()
#         self.sigma_1 = sigma_1

