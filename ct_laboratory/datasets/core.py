from ..samplers import Sampler
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Dict
import matplotlib.pyplot as plt
from pathlib import Path
import yaml


class GMI_Dataset(torch.utils.data.Dataset, ABC):
    """
    Abstract base class for all GMI datasets.
    All datasets in the GMI package should inherit from this class.
    """
    
    # Valid dataset types
    VALID_DATASET_TYPES = ['MNIST', 'MedMNIST']
    
    # Valid MedMNIST variants
    VALID_MEDMNIST_VARIANTS = [
        'PathMNIST', 'ChestMNIST', 'DermaMNIST', 'OCTMNIST', 'PneumoniaMNIST',
        'RetinaMNIST', 'BreastMNIST', 'BloodMNIST', 'TissueMNIST', 'OrganAMNIST',
        'OrganCMNIST', 'OrganSMNIST', 'OrganMNIST3D', 'NoduleMNIST3D',
        'AdrenalMNIST3D', 'FractureMNIST3D', 'VesselMNIST3D', 'SynapseMNIST3D'
    ]
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get a sample from the dataset."""
        pass
    
    def visualize(self, save_path: Optional[str] = None, num_samples: int = 100) -> None:
        """
        Visualize the dataset. This method should be implemented by subclasses.
        
        Args:
            save_path: Optional path to save the visualization
            num_samples: Number of samples to visualize
        """
        raise NotImplementedError(f"Visualization not implemented for {self.__class__.__name__}")
    
    @classmethod
    def from_config(cls, config_path: str) -> 'GMI_Dataset':
        """
        Create a dataset from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            GMI_Dataset instance
            
        Raises:
            ValueError: If the YAML configuration is invalid
            FileNotFoundError: If the YAML file doesn't exist
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"YAML configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['dataset_name', 'dataset_type']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in YAML configuration")
        
        dataset_name = config['dataset_name']
        dataset_type = config['dataset_type']
        
        # Validate dataset type
        if dataset_type not in cls.VALID_DATASET_TYPES:
            raise ValueError(f"Invalid dataset_type '{dataset_type}'. Valid types: {cls.VALID_DATASET_TYPES}")
        
        # Import dataset classes
        from .mnist import MNIST
        from .medmnist import MedMNIST
        
        # Create dataset based on type
        if dataset_type == 'MNIST':
            # Validate MNIST-specific fields
            if 'train' not in config:
                config['train'] = True
            if 'download' not in config:
                config['download'] = True
            if 'images_only' not in config:
                config['images_only'] = False
            
            return MNIST(
                train=config['train'],
                download=config['download'],
                images_only=config['images_only'],
                root=config.get('root')
            )
        
        elif dataset_type == 'MedMNIST':
            # Validate MedMNIST-specific fields
            if 'medmnist_name' not in config:
                raise ValueError("MedMNIST dataset requires 'medmnist_name' field")
            
            medmnist_name = config['medmnist_name']
            if medmnist_name not in cls.VALID_MEDMNIST_VARIANTS:
                raise ValueError(f"Invalid medmnist_name '{medmnist_name}'. Valid variants: {cls.VALID_MEDMNIST_VARIANTS}")
            
            # Validate other MedMNIST fields
            if 'split' not in config:
                config['split'] = 'train'
            if 'images_only' not in config:
                config['images_only'] = True
            
            return MedMNIST(
                dataset_name=medmnist_name,
                split=config['split'],
                images_only=config['images_only'],
                root=config.get('root')
            )
        
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")


class GeneralPurposeDataset(GMI_Dataset):
    """
    A general-purpose wrapper for external datasets that attempts to make them
    compatible with the GMI framework.
    """
    
    def __init__(self, dataset, expected_shape: Optional[Tuple[int, ...]] = None):
        """
        Initialize the wrapper.
        
        Args:
            dataset: The dataset to wrap
            expected_shape: Expected shape for samples (e.g., (1, H, W) for grayscale, (3, H, W) for RGB)
        """
        self.dataset = dataset
        self.expected_shape = expected_shape
        
        # Validate the dataset
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate that the dataset is compatible with our framework."""
        if not hasattr(self.dataset, '__len__'):
            raise ValueError("Dataset must have a __len__ method")
        
        if not hasattr(self.dataset, '__getitem__'):
            raise ValueError("Dataset must have a __getitem__ method")
        
        # Try to get a sample to check shape
        try:
            sample = self.dataset[0]
            if isinstance(sample, tuple):
                sample = sample[0]  # Assume first element is the image
            
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            
            # Check if it's a 2D image dataset
            if sample.dim() == 3:  # [C, H, W]
                if sample.shape[0] not in [1, 3]:
                    raise ValueError(f"Expected 1 or 3 channels, got {sample.shape[0]}")
            elif sample.dim() == 2:  # [H, W] - add channel dimension
                sample = sample.unsqueeze(0)
            else:
                raise ValueError(f"Expected 2D or 3D tensor, got {sample.dim()}D")
                
        except Exception as e:
            raise ValueError(f"Failed to validate dataset: {e}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Any:
        sample = self.dataset[idx]
        if isinstance(sample, tuple):
            return sample
        else:
            # Ensure proper tensor format
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            if sample.dim() == 2:
                sample = sample.unsqueeze(0)  # Add channel dimension
            return sample
    
    def visualize(self, save_path: Optional[str] = None, num_samples: int = 100) -> None:
        """
        Default visualization for general-purpose datasets.
        Creates a 10x10 grid showing samples from the dataset.
        """
        samples = []
        labels = []
        
        # Collect samples
        for i in range(min(num_samples, len(self))):
            try:
                data = self[i]
                if isinstance(data, tuple):
                    sample, label = data
                    labels.append(label)
                else:
                    sample = data
                    labels.append(None)
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Could not load sample {i}: {e}")
                continue
        
        if not samples:
            raise RuntimeError("No samples could be loaded for visualization")
        
        # Convert to tensor if needed
        if not isinstance(samples[0], torch.Tensor):
            samples = [torch.tensor(sample) for sample in samples]
        
        # Stack samples
        samples_tensor = torch.stack(samples)
        
        # Handle different tensor shapes
        if samples_tensor.dim() == 4:  # [batch, channels, height, width]
            if samples_tensor.shape[1] == 1:  # Grayscale
                samples_tensor = samples_tensor.squeeze(1)  # Remove channel dimension
            elif samples_tensor.shape[1] == 3:  # RGB
                samples_tensor = samples_tensor.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        # Create 10x10 visualization
        fig, axes = plt.subplots(10, 10, figsize=(20, 20))
        axes = axes.flatten()
        
        for i, (sample, ax) in enumerate(zip(samples_tensor, axes)):
            if sample.dim() == 2:  # Grayscale
                ax.imshow(sample.numpy(), cmap='gray')
            else:  # RGB or other
                ax.imshow(sample.numpy())
            
            if labels[i] is not None:
                ax.set_title(f'{labels[i]}', fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f'Dataset Visualization ({len(samples)} samples)', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()


class Medmnist_OrganA(Sampler):
    def __init__(self, 
                 root, 
                 train=True,
                 device='cpu'):
        # Load in the data
        if train:
            self.images = torch.from_numpy(np.load(root + '/organamnist_train_images.npy'))
            self.labels = torch.from_numpy(np.load(root + '/organamnist_train_labels.npy'))
        else:
            self.images = torch.from_numpy(np.load(root + '/organamnist_test_images.npy'))
            self.labels = torch.from_numpy(np.load(root + '/organamnist_test_labels.npy'))

        # Add an extra dimension for channels on axis 1
        self.images = torch.unsqueeze(self.images, 1)

        # Convert to float
        self.images = self.images.float()

        # Rescale to mean 0 and std 1
        self.mu = self.images.mean()
        self.sigma = self.images.std()
        self.images = (self.images - self.mu) / self.sigma

        # Move to device
        self.to(device)

        super(Medmnist_OrganA, self).__init__()

    def to(self, device):
        self.images = self.images.to(device)
        self.labels = self.labels.to(device)
        return self

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def sample(self, batch_size=1, return_labels=False):
        indices = torch.randint(0, len(self.images), (batch_size,))
        if return_labels:
            return self.images[indices], self.labels[indices]
        else:
            return self.images[indices]










class TCGA(Sampler):
    def __init__(self, 
                 root, 
                 train=True,
                 device='cpu',
                 num_files=1,
                 verbose=False):
        # Load in the data
        if train:
            image_list = []
            for i in range(num_files):
                if verbose:
                    print(f'Loading {root}/training/training_TCGA_LIHC_{str(i).zfill(6)}.pt')

                image_list.append(torch.load(root + f'/training/training_TCGA_LIHC_' + str(i).zfill(6) + '.pt'))
            self.images = torch.cat(image_list)

        else:
            image_list = []
            for i in range(num_files):
                if verbose:
                    print(f'Loading {root}/testing/testingTCGA_LIHC_{str(i).zfill(6)}.pt')
                image_list.append(torch.load(root + f'/testing/testing_TCGA_LIHC_' + str(i).zfill(6) + '.pt'))
            self.images = torch.cat(image_list)

        # Add an extra dimension for channels on axis 1
        self.images = torch.unsqueeze(self.images, 1)

        # Convert to float
        self.images = self.images.float()

        # Rescale to mean 0 and std 1
        self.mu = self.images.mean()
        self.sigma = self.images.std()
        self.images = (self.images - self.mu) / self.sigma

        # define the device for output images
        self.device = device

        super(TCGA, self).__init__()

    def __len__(self):
        return len(self.images)
    
    # remove the parts that have labels
    def __getitem__(self, idx):
        return self.images[idx].to(self.device)
    
    def sample(self, batch_size=1):
        indices = torch.randint(0, len(self.images), (batch_size,))
        return self.images[indices].to(self.device)

    def to(self, device):
        self.device = device
        return self


