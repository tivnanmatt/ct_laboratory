"""
Pytest configuration and common fixtures for GMI tests.
"""
import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session")
def device():
    """Return the device to use for tests (CPU or CUDA if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for tests that should not use GPU."""
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_tensor_2d():
    """Create a sample 2D tensor for testing."""
    return torch.randn(3, 4)


@pytest.fixture
def sample_tensor_3d():
    """Create a sample 3D tensor for testing."""
    return torch.randn(2, 3, 4)


@pytest.fixture
def sample_tensor_4d():
    """Create a sample 4D tensor for testing (batch, channels, height, width)."""
    return torch.randn(2, 1, 28, 28)


@pytest.fixture
def sample_complex_tensor():
    """Create a sample complex tensor for testing."""
    return torch.randn(3, 4) + 1j * torch.randn(3, 4)


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images for testing."""
    return torch.randn(4, 1, 64, 64)


@pytest.fixture
def sample_measurement_batch():
    """Create a sample batch of measurements for testing."""
    return torch.randn(4, 1, 32, 32)


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock wandb to avoid actual logging during tests."""
    class MockWandb:
        def __init__(self):
            self.log_calls = []
            self.init_calls = []
            self.finish_calls = []
        
        def init(self, *args, **kwargs):
            self.init_calls.append((args, kwargs))
            return self
        
        def log(self, *args, **kwargs):
            self.log_calls.append((args, kwargs))
        
        def finish(self, *args, **kwargs):
            self.finish_calls.append((args, kwargs))
    
    mock_wandb_instance = MockWandb()
    monkeypatch.setattr("wandb.init", mock_wandb_instance.init)
    monkeypatch.setattr("wandb.log", mock_wandb_instance.log)
    monkeypatch.setattr("wandb.finish", mock_wandb_instance.finish)
    return mock_wandb_instance


@pytest.fixture
def mock_hydra(monkeypatch):
    """Mock hydra to avoid actual config loading during tests."""
    class MockHydra:
        def __init__(self):
            self.compose_calls = []
            self.initialize_calls = []
        
        def compose(self, *args, **kwargs):
            self.compose_calls.append((args, kwargs))
            return {}
        
        def initialize(self, *args, **kwargs):
            self.initialize_calls.append((args, kwargs))
            return self
    
    mock_hydra_instance = MockHydra()
    monkeypatch.setattr("hydra.compose", mock_hydra_instance.compose)
    monkeypatch.setattr("hydra.initialize", mock_hydra_instance.initialize)
    return mock_hydra_instance


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary for testing."""
    return {
        "network": {
            "_target_": "gmi.network.unet.UNet",
            "in_channels": 1,
            "out_channels": 1,
            "model_channels": 64,
            "num_res_blocks": 2,
            "attention_resolutions": [8, 16],
            "dropout": 0.1,
            "channel_mult": [1, 2, 4],
            "conv_resample": True,
            "use_checkpoint": False,
            "use_fp16": False,
            "num_heads": 4,
            "num_head_channels": 32,
            "num_heads_upsample": -1,
            "use_scale_shift_norm": True,
            "resblock_updown": False,
            "use_new_attention_order": False,
        },
        "scheduler": {
            "_target_": "gmi.lr_scheduler.constant.ConstantScheduler",
            "learning_rate": 1e-4,
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 1e-4,
            "weight_decay": 0.01,
        }
    }


@pytest.fixture
def sample_vector_2():
    """Create a sample 2-element vector for 2x2 operator tests."""
    return torch.tensor([1.0, 2.0], dtype=torch.float32)


@pytest.fixture
def sample_matrix_2x2():
    """Create a sample 2x2 matrix for operator tests."""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32) 