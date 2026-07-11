import torch
from torch import nn
from .sde import StochasticDifferentialEquation

class UnconditionalDiffusionModel:
    def __init__(self, sde, nn_backbone, backbone_type='score'):
        
        assert isinstance(sde, StochasticDifferentialEquation)
        assert isinstance(nn_backbone, torch.nn.Module)
        assert backbone_type in ['score', 'mean', 'pushback']
        
        self.sde = sde
        self.nn_backbone = nn_backbone
        self.backbone_type = backbone_type

    def sample(self, x0, timesteps, sampler='euler', return_all=False):
        return self.sde.sample(x0, timesteps, sampler, return_all)
    
    def loss_closure(self, loss_fn):
        class LossClosure(nn.Module):
            def __init__(self, parent, model, loss_fn):
                super(LossClosure, self).__init__()
                self.parent = parent  # reference to the parent class
                self.model = model
                self.loss_fn = loss_fn

            def forward(self, batch_data):
                images = batch_data
                assert isinstance(self.parent, UnconditionalDiffusionModel)
                measurements = self.parent.sample_measurements_given_images(1, images)[0]  # reference parent method
                reconstructions = self.parent.sample_reconstructions_given_measurements(1, measurements)[0]  # reference parent method
                loss = self.loss_fn(reconstructions, images)
                return loss
                
        return LossClosure(self, self.image_reconstructor, loss_fn) 

