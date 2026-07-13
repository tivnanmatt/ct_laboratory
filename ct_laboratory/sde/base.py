import torch
import torch.nn as nn
from ..linear_system import LinearSystem


class StochasticDifferentialEquation(nn.Module):
    def __init__(self, f, G):
        """
        This class implements an Ito stochastic differential equation (SDE) of the form 
        
        dx = f(x, t) dt + G(x, t) dw
        
        f is a vector-valued function of x and t representing the drift term 
        and G is a matrix-valued function of x and t representing the diffusion rate.

        parameters:
            f: callable
                The drift term of the SDE. It should take x and t as input and return a tensor of the same shape as x.
            G: callable
                The diffusion term of the SDE. It should take x and t as input and return a gmi.linear_system.LinearSystem that can act on a tensor of the same shape as x.
        """

        super(StochasticDifferentialEquation, self).__init__()

        self.f = f
        self.G = G

        self.x_shape = None

    def forward(self, x, t):
        assert isinstance(x, torch.Tensor), "x must be a tensor."
        assert isinstance(t, torch.Tensor), "t must be a tensor."
        self.x_shape = x.shape
        return self.f(x, t), self.G(x, t)
    
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

        def compute_divergence_fn(GG_T, x):
            x_flattened = x.view(-1)  # Flatten the x tensor
            div = torch.zeros_like(x_flattened)
            for i in range(x_flattened.shape[0]):
                unit_vector = torch.zeros_like(x_flattened)
                unit_vector[i] = 1.0
                GG_T_unit = GG_T(unit_vector.view_as(x)).view(-1)
                try:
                    grad = torch.autograd.grad(GG_T_unit.sum(), x, retain_graph=True, create_graph=True)[0]
                    div[i] = grad.view(-1)[i]
                except RuntimeError:
                    # If gradient computation fails, set divergence to zero
                    div[i] = 0.0
            return div.view_as(x)  # Reshape the divergence to the original shape of x

        def _f_star(x, t):
            G_t = _G(x, t)
            G_tT = G_t.transpose_LinearSystem()
            GG_T = lambda v: G_t(G_tT(v))  # Define GG_T as a function to apply G_t and its transpose

            div_GG_T = compute_divergence_fn(GG_T, x)
            return _f(x, t) - div_GG_T - GG_T(score_estimator(x, t))

        return StochasticDifferentialEquation(f=_f_star, G=_G)

    def sample(self, x, timesteps, sampler='euler', return_all=False, verbose=False):
        """
        This method samples from the SDE.

        parameters:
            x: torch.Tensor
                The initial condition.
            timesteps: torch.Tensor
                The time steps at which the SDE is evaluated.
            sampler: str
                The method used to compute the forward update. Currently, only 'euler' and 'heun' are supported.
        returns:
            x: torch.Tensor
                The output tensor.
        """


        assert isinstance(x, torch.Tensor), "x must be a tensor."
        assert isinstance(timesteps, torch.Tensor), "timesteps must be a tensor."

        self.x_shape = x.shape

        # if timesteps is not a tensor, make it a tensor
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps)
    

        t_shape = [self.x_shape[0]] + [1]*(len(self.x_shape)-1) 
        _t = timesteps[0].reshape(1).repeat(self.x_shape[0]).reshape(t_shape)  # t should be [batch_size,*x_shape]

        if return_all:
            x_all = [x]
        
        for i in range(1, len(timesteps)):
            if verbose:
                print(f"Sampling step {i}/{len(timesteps)-1}")
                print(f"DEBUG: Memory usage: {torch.cuda.memory_allocated() / 1e9} GB")
            last_step = i == len(timesteps) - 1
            dt = timesteps[i] - _t
            x = self._sample_step(x, _t.view(-1), dt, sampler=sampler, last_step=last_step).detach()
            _t = timesteps[i].reshape(1).repeat(self.x_shape[0]).reshape(t_shape)

            if return_all:
                x_all.append(x)
        
        if return_all:
            return x_all
        
        return x

    def _sample_step(self, x, t, dt, sampler='euler', last_step=False):
        """
        This method computes the forward update of the SDE.

        The forward SDE is given by

        dx = f(x, t) dt + G(x, t) dw

        parameters:
            x: torch.Tensor
                The input tensor.
            t: float
                The time at which the SDE is evaluated.
            dt: float or torch.Tensor
                The time step.
            sampler: str
                The method used to compute the forward update. Currently, 'euler' and 'heun' are supported.
        returns:
            dx: torch.Tensor
                The output tensor.
        """

        if sampler == 'euler':
            return self._sample_step_euler(x, t, dt, last_step=last_step)
        elif sampler == 'heun':
            return self._sample_step_heun(x, t, dt, last_step=last_step)
        else:
            raise ValueError("The sampler should be one of ['euler', 'heun'].")

    def _sample_step_euler(self, x, t, dt, last_step=False):
        """
        This method computes the forward update of the SDE using the Euler-Maruyama method.

        The forward SDE is given by

        dx = f(x, t) dt + G(x, t) dw

        parameters:
            x: torch.Tensor
                The input tensor.
            t: float
                The time at which the SDE is evaluated.
            dt: float or torch.Tensor
                The time step.
            dw: torch.Tensor
                The Wiener process increment.
        returns:
            dx: torch.Tensor
                The output tensor.
        """

        if isinstance(dt, float):
            dt = torch.tensor(dt)

        dw = torch.randn_like(x) * torch.sqrt(torch.abs(dt))

        _f = self.f(x, t)
        assert isinstance(_f, torch.Tensor), "The drift term f(x, t) should return a tensor."
        assert _f.shape == x.shape, "The drift term f(x, t) should return a tensor of the same shape as x."
        
        _G = self.G(x, t)
        assert isinstance(_G, LinearSystem), "The diffusion term G(x, t) should return a LinearSystem."
        
        _f_dt = _f * dt
        _G_dw = _G.forward(dw)

        if last_step:
            return x + _f_dt
        else:
            return x + _f_dt + _G_dw

    def _sample_step_heun(self, x, t, dt, last_step=False):
        """
        This method computes the forward update of the SDE using the Heun's method.

        The forward SDE is given by

        dx = f(x, t) dt + G(x, t) dw

        parameters:
            x: torch.Tensor
                The input tensor.
            t: float
                The time at which the SDE is evaluated.
            dt: float or torch.Tensor
                The time step.
        returns:
            dx: torch.Tensor
                The output tensor.
        """

        if isinstance(dt, float):
            dt = torch.tensor(dt)

        dw = torch.randn_like(x) * torch.sqrt(dt)

        # Predictor step using Euler-Maruyama
        f_t = self.f(x, t)
        G_t = self.G(x, t)
        x_predict = x + f_t * dt + G_t.forward(dw)

        # Corrector step
        f_t_corrector = self.f(x_predict, t + dt)
        G_t_corrector = self.G(x_predict, t + dt)

        f_avg = (f_t + f_t_corrector) / 2
        G_dw_avg = (G_t.forward(dw) + G_t_corrector.forward(dw)) / 2


        if last_step:
            return x + f_avg * dt
        else:
            return x + f_avg * dt + G_dw_avg 