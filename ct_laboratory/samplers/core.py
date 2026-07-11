
import torch
from torch import nn

class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()

    def sample(self, batch_size, *args, **kwargs):
        raise NotImplementedError
    
    def forward(self, batch_size, *args, **kwargs):
        return self.sample(batch_size, *args, **kwargs)