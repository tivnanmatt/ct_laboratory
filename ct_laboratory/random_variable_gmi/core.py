
import torch
from torch import nn

from ..samplers import Sampler
from ..sde import StochasticDifferentialEquation
from ..linear_system import Identity

class RandomVariable(Sampler, torch.distributions.Distribution):
    def __init__(self):
        super(RandomVariable, self).__init__()

    def train_loss_closure(self, batch):
        self.train()
        return -self.log_prob(batch).mean()
    
    def eval_loss_closure(self, batch):
        self.eval()
        with torch.no_grad():
            return -self.log_prob(batch).mean()
    
    def log_prob(self, x):
        raise NotImplementedError
    
    def score(self, x):
        return torch.autograd.grad(self.log_prob(x), x, create_graph=True)
    
    def langevin_sde(self):
       _f = lambda x, t: -0.5 * self.score(x)
       _G = lambda x, t: Identity()
       return StochasticDifferentialEquation(_f, _G)


