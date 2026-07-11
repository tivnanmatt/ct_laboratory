from .scalar_sde import ScalarSDE


class VarianceExplodingSDE(ScalarSDE):
    def __init__(self, noise_variance, noise_variance_prime=None):
        """
        This class implements a variance-exploding process, which is a mean-reverting process with a variance-exploding term.
        
        parameters:
            sigma_1: float
                The standard deviation at t=1 (the variance at t=1 is G*G^T = sigma_1^2)
        """
        signal_scale = lambda t: 1.0
        signal_scale_prime = lambda t: 0.0

        super(VarianceExplodingSDE, self).__init__(signal_scale=signal_scale, noise_variance=noise_variance, signal_scale_prime=signal_scale_prime, noise_variance_prime=noise_variance_prime) 