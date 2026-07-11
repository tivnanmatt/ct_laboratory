

import torch
from . import Sampler


class ModuleSampler(Sampler):
    def __init__(self, module):
        super(ModuleSampler, self).__init__()
        assert isinstance(module, torch.nn.Module), 'module must be a torch.nn.Module object'
        self.module = module
    
    def sample(self, batch_size, *args, **kwargs):
        for i in range(batch_size):
            _sample = self.module(*args, **kwargs)
            if i == 0:
                assert isinstance(_sample, torch.Tensor), 'module must return a torch.Tensor object'
                sample_shape = _sample.shape
                samples = torch.zeros(batch_size, *sample_shape)
            samples[i] = _sample
        return samples


