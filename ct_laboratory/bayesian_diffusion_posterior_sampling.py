import torch
from torch import nn
import time

class BayesianDiffusionPosteriorSampling:
    """
    Class to perform MAP reconstruction given a log-likelihood and log-prior function.
    """
    def __init__(
        self,
        log_likelihood_fn,
        bayesian_denoiser,
        noise_var_schedule,
        volume_init,
        num_iterations_map=50,
        regularization_weight=1.0,
        preconditioner=None,
        inv_preconditioner=None,
        lr=1e-3,
        warmup_iters=None,
        randomize_likelihood_every=None,
        print_every=1,
        verbose=False
    ):
        self.log_likelihood_fn = log_likelihood_fn
        self.bayesian_denoiser = bayesian_denoiser
        self.noise_var_schedule = noise_var_schedule

        # these are related to the map reconstruction step
        self.regularization_weight = regularization_weight
        self.preconditioner = preconditioner
        self.inv_preconditioner = inv_preconditioner
        self.lr = lr
        self.num_iterations_map = num_iterations_map
        self.warmup_iters = warmup_iters
        self.randomize_likelihood_every = randomize_likelihood_every
        self.print_every = print_every
        self.verbose = verbose

        self.volume_pred = volume_init.clone().detach()
        self.volume_t = self.volume_pred + torch.randn_like(self.volume_pred) * torch.sqrt(self.noise_var_schedule[0])
        
        # Initialize optimizer and preconditioned volume
        if self.preconditioner is not None:
            self.volume_pred_preconditioned = self.inv_preconditioner(self.volume_pred)
            self.volume_pred_preconditioned = self.volume_pred_preconditioned.clone().detach().requires_grad_(True)
            self.optimizer = torch.optim.Adam([self.volume_pred_preconditioned], lr=lr)
        else:
            self.volume_pred = self.volume_pred.clone().detach().requires_grad_(True)
            self.optimizer = torch.optim.Adam([self.volume_pred], lr=lr)

        if warmup_iters is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda it: min(1.0, float(it) / warmup_iters)
            )
        else:
            self.scheduler = None

    def map_step(self, log_prior_fn):
        """Perform a single MAP optimization step."""
        self.optimizer.zero_grad()
        if self.preconditioner is not None:
            self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)
        log_likelihood = self.log_likelihood_fn(self.volume_pred)
        log_prior = self.regularization_weight * log_prior_fn(self.volume_pred)
        log_posterior = log_likelihood + log_prior
        loss = -1.0 * log_posterior
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.preconditioner is not None:
            self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)
        return log_likelihood.item(), log_prior.item(), log_posterior.item()

    def diffusion_posterior_sampling_step(self, iteration):

        var = self.noise_var_schedule[iteration]
        delta_var = var - self.noise_var_schedule[iteration + 1]
        
        denoiser_mu, denoiser_log_var = self.bayesian_denoiser(self.volume_t, var)
        denoiser_var = torch.exp(denoiser_log_var)

        def log_prior_fn(volume):
            log_prior = -0.5 * (((volume - denoiser_mu) ** 2 / denoiser_var).sum())
            return log_prior
        
        # Reinitialize volume_pred to current volume_t for this diffusion step
        # self.volume_pred = self.volume_t.clone().detach()
        if self.preconditioner is not None:
            self.volume_pred_preconditioned = self.inv_preconditioner(self.volume_pred)
            self.volume_pred_preconditioned = self.volume_pred_preconditioned.clone().detach().requires_grad_(True)
            # Update optimizer parameters
            self.optimizer.param_groups[0]['params'] = [self.volume_pred_preconditioned]
        else:
            self.volume_pred = self.volume_pred.clone().detach().requires_grad_(True)
            # Update optimizer parameters
            self.optimizer.param_groups[0]['params'] = [self.volume_pred]
        
        # Reset scheduler if using warmup
        if self.warmup_iters is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda it: min(1.0, float(it) / self.warmup_iters)
            )
        
        # Run MAP reconstruction
        t0 = time.time()
        if self.verbose:
            print(f"Starting MAP reconstruction")
        for it in range(1, self.num_iterations_map + 1):
            
            if self.randomize_likelihood_every is not None:
                if it > 0 and it % self.randomize_likelihood_every == 0:
                    self.log_likelihood_fn.randomize()

            log_likelihood, log_prior, log_posterior = self.map_step(log_prior_fn)
            
            if self.verbose:
                if it % self.print_every == 0 or it == 1:
                    dt = time.time() - t0
                    print(f"Iter {it:04d}  likelihood={-1.0*log_likelihood:.4e}  prior={-1.0*log_prior:.4e} posterior={-1.0*log_posterior:.4e}   ({dt:.2f} sec)")
                    t0 = time.time()

        # score = (1/var) * (self.volume_pred - self.volume_t)

        # self.volume_t = self.volume_t + 0.5*delta_var*score


        self.volume_t = self.volume_pred + torch.randn_like(self.volume_pred) * torch.sqrt(self.noise_var_schedule[iteration + 1])  

        # epsilon = torch.randn_like(self.volume_t)
        # self.volume_t = self.volume_t + 0.5*delta_var*score + torch.sqrt(delta_var) * epsilon

    def run_diffusion_posterior_sampling(self):
        num_diffusion_steps = len(self.noise_var_schedule) - 1
        for it in range(num_diffusion_steps):
            if self.verbose:
                print(f"Diffusion step {it+1}/{num_diffusion_steps} with noise var {self.noise_var_schedule[it]:.4e} -> {self.noise_var_schedule[it+1]:.4e}")
            self.diffusion_posterior_sampling_step(it)
        return self.volume_t    