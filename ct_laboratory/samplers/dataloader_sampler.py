

import torch
from . import Sampler


class DataLoaderSampler(Sampler):
    def __init__(self, dataloader):
        super(DataLoaderSampler, self).__init__()
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'dataloader must be a torch.utils.data.DataLoader object'
        self.dataloader = dataloader
    
    def sample(self, batch_size):
        samples = next(iter(self.dataloader))
        assert samples.shape[0] == batch_size, 'batch_size must be equal to the batch size of the dataloader'
        return samples