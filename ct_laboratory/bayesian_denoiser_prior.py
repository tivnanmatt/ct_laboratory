
import torch
import torch.nn as nn

class BayesianDenoiserPrior(nn.Module):
    """
    Bayesian Denoiser Prior using a learned Bayesian denoiser.
    The prior log-probability is computed based on the denoiser's posterior estimates.
    """
    def __init__(self, bayesian_denoiser, regularization_weight=1.0, noise_var=0.01):
        self.bayesian_denoiser = bayesian_denoiser
        self.regularization_weight = regularization_weight

        if isinstance(noise_var, float) or isinstance(noise_var, int):
            noise_var = torch.tensor([noise_var])
        elif isinstance(noise_var, torch.Tensor):
            if noise_var.dim() == 0:
                noise_var = torch.tensor([noise_var.item()])
            self.noise_var_schedule = noise_var
            self.noise_var = self.noise_var_schedule[0]
            self.schedule_counter = 0
        super().__init__()

    def forward(self, volume: torch.Tensor, noise_var=None) -> torch.Tensor:
        """
        volume: (n_x, n_y, n_z) or (batch, n_x, n_y, n_z).
        For simplicity assume shape is (n_x, n_y, n_z). If batch is
        used, adapt the shift dims accordingly.
        """
        if noise_var is not None:
            self.noise_var = noise_var
        else:
            self.noise_var = self.noise_var_schedule[self.schedule_counter]
            self.schedule_counter = min(self.schedule_counter + 1, len(self.noise_var_schedule) - 1)
        
        # Ensure noise_var is a tensor
        if not isinstance(self.noise_var, torch.Tensor):
            noise_var_tensor = torch.tensor(self.noise_var, dtype=volume.dtype, device=volume.device)
        else:
            noise_var_tensor = self.noise_var
        
        noisy_volume = volume + torch.randn_like(volume) * torch.sqrt(noise_var_tensor)

        # denoised_mu = self.normal_diffusion_denoiser(noisy_volume, noise_var_tensor)

        # denoised_mu = self.normal_time_based_diffusion_denoiser(volume_at_time_t, time_t)
        # score_function = ... (mu)
        # dx/dt = ... (score_function)
        # x' = x + dx/dt * dt

        denoised_mu, denoised_log_var = self.bayesian_denoiser(noisy_volume, noise_var_tensor)
        
        # Compute constant term using tensors
        # two_pi = torch.tensor(2.0 * 3.14159265359, dtype=volume.dtype, device=volume.device)
        # const = -0.5 * torch.log(two_pi) * volume.numel() - 0.5 * denoised_log_var.sum()
        const = 0.0  # Constant term can be ignored in most cases
        
        log_prior = -0.5 * (((volume - denoised_mu) ** 2 / torch.exp(denoised_log_var)).sum()) + const
        return self.regularization_weight * log_prior
