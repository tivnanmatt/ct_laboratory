import torch
from . import Sampler

class DatasetSampler(Sampler):
    def __init__(self, dataset, batch_size=1):
        super(DatasetSampler, self).__init__()
        assert isinstance(dataset, torch.utils.data.Dataset), 'dataset must be a torch.utils.data.Dataset object'
        self.dataset = dataset
        self.batch_size = batch_size
    
    def sample(self, batch_size):
        self.batch_size = batch_size
        indices = torch.zeros(self.batch_size, dtype=torch.long)
        for i in range(self.batch_size):
            indices[i] = torch.randint(0, len(self.dataset), (1,))
            while indices[i] in indices[:i]:
                indices[i] = torch.randint(0, len(self.dataset), (1,))
        print(f"[DatasetSampler] Sampled indices: {indices.tolist()}")
        dataset_samples = [self.dataset[i] for i in indices]
        print(f"[DatasetSampler] Sampled values: {dataset_samples}")
        data_batch = torch.stack(dataset_samples)
        return data_batch
    


