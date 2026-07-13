import torch
from .scalar_sde import ScalarSDE


class VariancePreservingSDE(ScalarSDE):
    def __init__(self, beta=5.0):
        """
        This class implements a variance-preserving process, which is a mean-reverting process with a variance-preserving term.
        
        parameters:
            beta: float
                The variance-preserving coefficient.
        """

        if isinstance(beta, float):
            beta = torch.tensor(beta)

        signal_scale = lambda t: torch.exp(-0.5*beta*t)
        signal_scale_prime = lambda t: -0.5*beta*torch.exp(-0.5*beta*t)
        noise_variance = lambda t: beta*t
        noise_variance_prime = lambda t: beta

        super(VariancePreservingSDE, self).__init__(signal_scale=signal_scale, noise_variance=noise_variance, signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime) 