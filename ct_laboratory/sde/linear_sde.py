import torch
from .base import StochasticDifferentialEquation
from ..linear_system import InvertibleLinearSystem, SymmetricLinearSystem


class LinearSDE(StochasticDifferentialEquation):
    def __init__(self, H, Sigma, H_prime=None, Sigma_prime=None, F=None, G=None):
        """
        This class implements a linear stochastic differential equation (SDE) of the form:
        dx = F(t) @ x dt + G(t) dw
        where F and G are derived from H and Sigma if not directly provided.

        Parameters:
            H: callable
                Function that returns an InvertibleLinearSystem representing the system response.
            Sigma: callable
                Function that returns a SymmetricLinearSystem representing the covariance.
            H_prime: callable, optional
                Function that returns the time derivative of H. If not provided, it will be computed automatically.
            Sigma_prime: callable, optional
                Function that returns the time derivative of Sigma. If not provided, it will be computed automatically.
            F: callable, optional
                Function that returns a LinearSystem representing the drift term. If not provided, it will be computed from H_prime and H.
            G: callable, optional
                Function that returns a LinearSystem representing the diffusion term. If not provided, it will be computed from Sigma_prime, F, and Sigma.

        Requirements:
            - H must return an InvertibleLinearSystem.
            - Sigma must return a SymmetricLinearSystem.
            - The @ operator must be implemented for matrix-matrix multiplication of F, Sigma, and their transposes.
            - The addition, subtraction, and sqrt_LinearSystem methods must be implemented for the resulting matrix operations on Sigma_prime and others.

        If H_prime and Sigma_prime are not provided, they will be computed using automatic differentiation.
        """

        assert isinstance(H(0.0), InvertibleLinearSystem), "H(t) must return an InvertibleLinearSystem."
        assert isinstance(Sigma(0.0), SymmetricLinearSystem), "Sigma(t) must return a SymmetricLinearSystem."

        self.H = H
        self.Sigma = Sigma
        self.H_prime = H_prime
        self.Sigma_prime = Sigma_prime
        self.F = F
        self._G = G

        assert H_prime is not None or F is not None, "Either H_prime or F must be provided."
        assert Sigma_prime is not None or G is not None, "Either Sigma_prime or G must be provided."

        if F is None and H_prime is not None:
            self.F = lambda t: self.H_prime(t) @ self.H(t).inverse_LinearSystem()

        if self._G is None and Sigma_prime is not None:
            self._G = lambda t: (self.Sigma_prime(t) - self.F(t) @ self.Sigma(t) - self.Sigma(t) @ self.F(t).transpose_LinearSystem()).sqrt_LinearSystem()

        _f = lambda x, t: self.F(t).forward(x)
        _G = lambda x, t: self._G(t)

        super(LinearSDE, self).__init__(f=_f, G=_G)
        
    def reverse_SDE_given_score_estimator(self, score_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a score function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - div_x( G(x,t) G(x,t)^T ) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            score_estimator: callable
                The score estimator function that takes x, t, as input and returns the score function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """
        _f = self.f
        _G = self.G

        def compute_divergence(GG_T, x):
            # divergence always zero for operators that do not depend on x
            div = torch.zeros_like(x)
            return div  

        def _f_star(x, t):
            G_t = _G(x, t)
            G_tT = G_t.transpose_LinearSystem()
            GG_T = lambda v: G_t(G_tT(v))  # Define GG_T as a function to apply G_t and its transpose

            div_GG_T = compute_divergence(GG_T, x)
            return _f(x, t) - div_GG_T - GG_T(score_estimator(x, t))

        return StochasticDifferentialEquation(f=_f_star, G=_G)
    
    def reverse_SDE_given_mean_estimator(self, mean_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a mean function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            mean_estimator: callable
                The mean estimator function that takes x, t, as input and returns the mean function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """
        
        def score_estimator(x, t):

            assert isinstance(x, torch.Tensor), "x must be a tensor."

            Sigma_t = self.Sigma(t)

            assert isinstance(Sigma_t, InvertibleLinearSystem), "Sigma(t) must be an InvertibleLinearSystem."

            Sigma_t_inv = self.Sigma(t).inverse_LinearSystem()

            mu_t = mean_estimator(x, t)

            return Sigma_t_inv.forward(mu_t-x)

        return self.reverse_SDE_given_score_estimator(score_estimator)

    def reverse_SDE_given_noise_estimator(self, noise_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a noise function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            noise_estimator: callable
                The noise estimator function that takes x, t, as input and returns the noise function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """

        # score = Sigma(t)^(-1) @ (x - mu_t)
        # x = mu_t + Sigma(t)^(1/2) @ noise
        # noise = Sigma(t)^(-1/2) @ (x - mu_t)
        # score = Sigma(t)^(-1) @ (mu_t + Sigma(t)^(1/2) @ noise - mu_t)
        # score = Sigma(t)^(-1) @ Sigma(t)^(1/2) @ noise
        # score = Sigma(t)^(-1/2) @ noise
        
        def score_estimator(x, t):
            noise_t = noise_estimator(x, t)
            sigma_t_sqrt_inv = self.Sigma(t).sqrt_LinearSystem().inverse_LinearSystem()
            return -1.0*(sigma_t_sqrt_inv.forward(noise_t))

        return self.reverse_SDE_given_score_estimator(score_estimator)
    
    def mean_response_x_t_given_x_0(self, x0, t):
        """
        Computes the mean response of x_t given x_0.

        Parameters:
            x0: torch.Tensor
                The initial condition.
            t: float
                The time at which the mean response is evaluated.
        
        Returns:
            torch.Tensor
                The mean response at time t.
        """

        assert isinstance(x0, torch.Tensor), "x0 must be a tensor."
        assert isinstance(t, torch.Tensor), "t must be a tensor."

        self.x_shape = x0.shape

        return self.H(t).forward(x0)

    def sample_x_t_given_x_0(self, x0, t):
        """
        Samples x_t given x_0 using the mean response and adding Gaussian noise with covariance Sigma(t).

        Parameters:
            x0: torch.Tensor
                The initial condition.
            t: float
                The time at which the sample is evaluated.
        
        Returns:
            torch.Tensor
                The sampled response at time t.
        """

        assert isinstance(x0, torch.Tensor), "x0 must be a tensor."
        assert isinstance(t, torch.Tensor), "t must be a tensor."

        self.x_shape = x0.shape
        
        noise = torch.randn_like(x0)
        return self.sample_x_t_given_x_0_and_noise(x0, noise, t)

    def sample_x_t_given_x_0_and_noise(self, x0, noise, t):
        """
        Samples x_t given x_0 using the mean response and adding Gaussian noise with covariance Sigma(t).

        Parameters:
            x0: torch.Tensor
                The initial condition.
            t: float
                The time at which the sample is evaluated.
        
        Returns:
            torch.Tensor
                The sampled response at time t.
        """
        assert isinstance(x0, torch.Tensor), "x0 must be a tensor."
        assert isinstance(t, torch.Tensor), "t must be a tensor."

        self.x_shape = x0.shape

        mean_response = self.mean_response_x_t_given_x_0(x0, t)
        Sigma_sqrtm = self.Sigma(t).sqrt_LinearSystem()
        return mean_response + Sigma_sqrtm.forward(noise)

    def reverse_SDE_given_posterior_mean_estimator(self, posterior_mean_estimator):
        """
        Constructs the reverse-time stochastic differential equation given a posterior mean estimator.

        The time-reversed SDE is given by:
        dx = f*(x, t) dt + G(x, t) dw
        where f*(x, t) = f(x, t) - G(x, t) G(x, t)^T score_estimator(x, t)
        and score_estimator(x, t) = Sigma(t)^(-1) @ (x - mu_t)

        Parameters:
            posterior_mean_estimator: callable
                Function that takes x and t as input and returns the estimated mean at time t.
        
        Returns:
            StochasticDifferentialEquation
                The reverse-time SDE.
        """
        
        def score_estimator(x, t):
            mu_t = posterior_mean_estimator(x, t)
            sigma_t_inv = self.Sigma(t).inverse_LinearSystem()
            return sigma_t_inv.forward(x - mu_t)

        return self.reverse_SDE_given_score_estimator(score_estimator) 