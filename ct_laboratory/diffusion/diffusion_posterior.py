import torch
from torch import nn
from ..sde import LinearSDE, StandardWienerSDE
from ..linear_system import DiagonalScalar
from ..random_variable_gmi import UniformRandomVariable, GaussianRandomVariable
from ..samplers import Sampler
from .core import DiffusionModel

class DiffusionPosteriorModel(DiffusionModel):
    def __init__(self,
                 diffusion_backbone,
                 forward_SDE=None,
                 training_loss_fn=None,
                 training_time_sampler=None,
                 training_time_uncertainty_sampler=None,
                 device=None):
        """
        This is an abstract base class for diffusion models.
        """
        torch.nn.Module.__init__(self)

        assert isinstance(diffusion_backbone, torch.nn.Module)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if forward_SDE is None:
            forward_SDE = StandardWienerSDE()

        if training_loss_fn is None:

            class KLDivergenceTrainingLossFn(torch.nn.Module):
                def __init__(self):
                    super(KLDivergenceTrainingLossFn, self).__init__()

                def forward(self, clean, mean_pred, logvar_pred, t):
                    var_pred = torch.exp(logvar_pred)
                    quad = (clean - mean_pred).pow(2) / var_pred
                    kl = 0.5 * (quad + logvar_pred)
                    loss = kl.mean()
                    return loss
                    
            training_loss_fn = KLDivergenceTrainingLossFn().to(device)

        if training_time_sampler is None:
            training_time_sampler = UniformRandomVariable(0.0, 1.0)

        if training_time_uncertainty_sampler is None:
            class IdentitySampler(Sampler):
                def sample(self, t):
                    return t
            training_time_uncertainty_sampler = IdentitySampler()

        assert isinstance(forward_SDE, LinearSDE)
        assert isinstance(diffusion_backbone, torch.nn.Module)
        assert isinstance(training_time_sampler, Sampler)
        assert isinstance(training_loss_fn, torch.nn.Module)
        assert isinstance(training_time_uncertainty_sampler, Sampler)

        self.diffusion_backbone = diffusion_backbone
        self.forward_SDE = forward_SDE
        self.training_loss_fn = training_loss_fn
        self.training_time_sampler = training_time_sampler
        self.training_time_uncertainty_sampler = training_time_uncertainty_sampler


    # use union of two data types for typing
    def forward(self, batch_data: torch.Tensor | list):
        """
        This method implements the training loss closure of the diffusion model.
        It computes the loss between the predicted x_0 and the true x_0.
        parameters:
            batch_data: torch.Tensor or list
                The input tensor to the linear operator.
                If it is a tensor, it is assumed to be the true x_0.
                If it is a list, it is assumed to be a tuple of (x_0, y).
        returns:
            loss: torch.Tensor
                The loss between the predicted x_0 and the true x_0.
        """

        if isinstance(batch_data, torch.Tensor):
            x_0 = batch_data
            y=None
        elif isinstance(batch_data, list):
            assert len(batch_data) == 2, "batch_data should a tensor (unconditional) or a tuple/list of two elements (conditional)"
            x_0 = batch_data[0]
            y = batch_data[1]
        else:
            raise ValueError("batch_data should a tensor (unconditional) or a tuple/list of two elements (conditional)")

        batch_size = x_0.shape[0]

        t = self.training_time_sampler.sample(batch_size).to(x_0.device)
        tau = self.training_time_uncertainty_sampler.sample(t).to(x_0.device)
        x_t = self.forward_SDE.sample_x_t_given_x_0(x_0, tau)
        mean_pred, logvar_pred = self.predict_mean_and_log_var(x_t, t, y)
        loss = self.training_loss_fn(x_0, mean_pred, logvar_pred, t)
        return loss
        
    def predict_x_0(self, x_t: torch.Tensor, t: torch.Tensor, y=None):
        """
        This method predicts x_0 given x_t.

        parameters:
            x_t: torch.Tensor
                The sample at time t.
            t: float
                The time step.
        returns:
            x_0: torch.Tensor
                The predicted initial condition.
        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)

        if y is None:
            mean_pred, _ =  self.diffusion_backbone(x_t, t)
        else:
            mean_pred, _ =  self.diffusion_backbone(x_t, t, y)

        return mean_pred
    

    def predict_mean_and_log_var(self, x_t: torch.Tensor, t: torch.Tensor, y=None):
        """
        This method predicts the mean and log variance of the posterior distribution.

        parameters:
            x_t: torch.Tensor
                The sample at time t.
            t: float
                The time step.
        returns:
            mean_pred: torch.Tensor
                The predicted mean of the posterior distribution.
            logvar_pred: torch.Tensor
                The predicted log variance of the posterior distribution.
        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)

        if y is None:
            mean_pred, logvar_pred =  self.diffusion_backbone(x_t, t)
        else:
            mean_pred, logvar_pred =  self.diffusion_backbone(x_t, t, y)

        return mean_pred, logvar_pred
    

    def sample_reverse_process_BDPS(self,    
                                x_t, 
                                log_likelihood_fn, 
                                timesteps, 
                                likelihood_weight=1.0,
                                jacobian_method='Backpropagation',  # or 'Identity'
                                sampler='euler', 
                                return_all=False, 
                                y=None, 
                                verbose=False):
        """
        Samples from the reverse SDE using diffusion posterior sampling (DPS).

        Parameters:
            x_t: torch.Tensor
                The initial condition.
            log_likelihood_fn: Callable
                Function that returns scalar log-likelihood given x_0.
            timesteps: int
                Number of timesteps to sample.
            likelihood_weight: float
                Weight on the likelihood score.
            jacobian_method: str
                One of ['Backpropagation', 'Identity']
            sampler: str
                Sampler method: 'euler' or 'heun'.
            return_all: bool
                If True, return all intermediate samples.
            y: torch.Tensor
                Optional conditional input.
        Returns:
            x: torch.Tensor
                Final output tensor.
        """

        def map_reconstructor(x_init, mu_pred, logvar_pred, lr=1e-1, n_steps=200):
            """
            This function reconstructs the posterior mean given the predicted mean and log variance.
            """

            mu_pred = mu_pred.detach()
            logvar_pred = logvar_pred.detach()
            prior_distribution = GaussianRandomVariable(mu_pred, DiagonalScalar(torch.exp(logvar_pred)))

            x_recon = x_init.detach().requires_grad_(True)
            optimizer = torch.optim.Adam([x_recon], lr=lr)
            for i_step in range(n_steps):
                optimizer.zero_grad()
                log_prior = prior_distribution.log_prob_plus_constant(x_recon)
                log_likelihood = log_likelihood_fn(x_recon)
                log_posterior = log_prior + likelihood_weight * log_likelihood
                negative_log_posterior = -log_posterior
                negative_log_posterior.backward()
                optimizer.step()
                # x_recon.data = x_recon.data.clamp(0.0, 1.0)
                # print(f"Step {i_step+1}/{n_steps}, negative_log_posterior: {negative_log_posterior.item()}, negative_log_prior: {-log_prior.item()}, negative_log_likelihood: {-log_likelihood.item()}", end='\r')
                # print(f"Step {i_step+1}/{n_steps}, negative_log_posterior: {negative_log_posterior.item()}") 
            return x_recon
            
        
        def posterior_mean_estimator(x_t, t, y=None):
            """
            This function estimates the posterior mean given x_t, t, and y.
            """

            if y is None:
                mean_pred, logvar_pred = self.predict_mean_and_log_var(x_t, t)
            else:
                mean_pred, logvar_pred = self.predict_mean_and_log_var(x_t, t, y)

            # reconstruct the posterior mean
            x_recon = map_reconstructor(mean_pred.clone(), mean_pred, logvar_pred)
            # x_recon = mean_pred
            return x_recon
        
        reverse_SDE = self.forward_SDE.reverse_SDE_given_mean_estimator(posterior_mean_estimator)
        return reverse_SDE.sample(x_t, timesteps, sampler, return_all, verbose)






    
class DiffusionBackbone(torch.nn.Module):
    def __init__(self,
                 x_t_encoder,
                 t_encoder,
                 x_0_predictor,
                 y_encoder=None,
                 pass_t_to_x_0_predictor=False,
                 c_skip=None,
                 c_out=None,
                 c_in=None,
                 c_noise=None):
        
        """
        
        This is designed to implement a diffusion backbone. It predicts x_0 given x_t, t, and y embeddings.

        x_t is a sample from the forward or reverse diffusion process at time t, it is a tensor of shape [batch_size, *x_t.shape]
        t is the time step. We assume it is a tensor of shape [batch_size, 1]
        y is an optional conditional input to the diffusion backbone, it is a tensor of shape [batch_size, *y.shape]


        parameters:
            x_t_encoder: torch.nn.Module
                The neural network that encodes information from x_t.
            t_encoder: torch.nn.Module
                The neural network that encodes information from t.
            x_0_predictor: torch.nn.Module
                The neural network that predicts x_0 given x_t, t, and y embeddings.
            y_encoder: torch.nn.Module
                The optional neural network that encodes information from y.
        """
        

        assert isinstance(x_t_encoder, torch.nn.Module)
        assert isinstance(t_encoder, torch.nn.Module)
        assert isinstance(x_0_predictor, torch.nn.Module)

        if y_encoder is not None:
            assert isinstance(y_encoder, torch.nn.Module)

        super(DiffusionBackbone, self).__init__()

        self.x_t_encoder = x_t_encoder
        self.t_encoder = t_encoder
        self.x_0_predictor = x_0_predictor
        self.y_encoder = y_encoder

        self.pass_t_to_x_0_predictor = pass_t_to_x_0_predictor

        def expand_t_to_x_shape(t, x_shape):
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == x_shape[0], f"t.shape[0] = {t.shape[0]}, x_shape[0] = {x_shape[0]}"
            t_shape = [t.shape[0]] + [1]*len(x_shape[1:])
            t = t.reshape(t_shape)
            return t

        # var_data = 0.25

        if c_skip is None:
            c_skip = lambda t,x_shape: 0.0
        
        if c_out is None:
            c_out = lambda t,x_shape: 1.0

        if c_in is None:
            c_in = lambda t,x_shape: 1.0
            
        if c_noise is None:
            c_noise = lambda t,x_shape: 1.0

        self.c_skip = c_skip
        self.c_out = c_out
        self.c_in = c_in
        self.c_noise = c_noise

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y=None):

        """
        This method implements the forward pass of the diffusion backbone.

        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)
        
        x_t_embedding = self.x_t_encoder(x_t)
        t_embedding = self.t_encoder(t)

        if y is not None:
                y_embedding = self.y_encoder(y)

        if self.pass_t_to_x_0_predictor:
            if y is not None:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, y_embedding, (self.c_noise(t,t.shape)*t).squeeze())
            else:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, (self.c_noise(t,t.shape)*t).squeeze())
        else:
            if y is not None:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, y_embedding)
            else:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding)


        x_0_pred = self.c_skip(t,x_t.shape)*x_t + self.c_out(t,x_out.shape)*x_out



        # if self.pass_t_to_x_0_predictor:
        #     if y is not None:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding, y_embedding, t.squeeze())
        #     else:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding, t.squeeze())
        # else:
        #     if y is not None:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding, y_embedding)
        #     else:
        #         x_out = self.x_0_predictor(x_t_embedding, t_embedding)


        # x_0_pred = x_out
        

        return x_0_pred



