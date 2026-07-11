from .variance_exploding_sde import VarianceExplodingSDE


class StandardWienerSDE(VarianceExplodingSDE):
    def __init__(self):
        """
        This class implements a Wiener process, which is a Song variance-exploding process with sigma_1 = 1.
        """

        noise_variance = lambda t: t
        noise_variance_prime = lambda t: 0*t + 1.0
        super(StandardWienerSDE, self).__init__(noise_variance=noise_variance,
                                            noise_variance_prime=noise_variance_prime) 