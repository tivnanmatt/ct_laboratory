"""
Command for training diffusion models from YAML configuration.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from ..diffusion.core import DiffusionModel
from ..config import load_and_merge_configs, load_object_from_dict


def train_diffusion_model_from_configs(config_paths: List[Union[str, Path]], 
                                     device: Optional[str] = None, 
                                     experiment_name: Optional[str] = None,
                                     train_dataset_path: Optional[str] = None,
                                     validation_dataset_path: Optional[str] = None,
                                     test_dataset_path: Optional[str] = None,
                                     diffusion_backbone_path: Optional[str] = None,
                                     output_dir: Optional[str] = None):
    """
    Train a diffusion model from multiple YAML configuration files.
    
    Args:
        config_paths: List of paths to YAML configuration files
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
        train_dataset_path: Optional path to override train dataset config
        validation_dataset_path: Optional path to override validation dataset config
        test_dataset_path: Optional path to override test dataset config
        diffusion_backbone_path: Optional path to override diffusion backbone config
        output_dir: Optional output directory (default: gmi_data/outputs/{experiment_name})
        
    Returns:
        Tuple of (train_losses, val_losses, eval_metrics)
    """
    # Load and merge all config files
    config_dict = load_and_merge_configs(config_paths)
    
    # Override components with specific config files if provided
    if train_dataset_path:
        train_config = load_and_merge_configs([train_dataset_path])
        # Map the component name to the standard key
        for key, value in train_config.items():
            if 'train_dataset' in key or 'dataset' in key:
                config_dict['train_dataset'] = value
                break
    
    if validation_dataset_path:
        val_config = load_and_merge_configs([validation_dataset_path])
        for key, value in val_config.items():
            if 'validation_dataset' in key:
                config_dict['validation_dataset'] = value
                break
    
    if test_dataset_path:
        test_config = load_and_merge_configs([test_dataset_path])
        for key, value in test_config.items():
            if 'test_dataset' in key:
                config_dict['test_dataset'] = value
                break
    
    if diffusion_backbone_path:
        backbone_config = load_and_merge_configs([diffusion_backbone_path])
        for key, value in backbone_config.items():
            if 'diffusion_backbone' in key or 'backbone' in key:
                config_dict['diffusion_backbone'] = value
                break
    
    return _train_from_config_dict(config_dict, device, experiment_name, output_dir)


def train_diffusion_model(config_path: str, device: Optional[str] = None, experiment_name: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Train a diffusion model from a single YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
        output_dir: Optional output directory (default: gmi_data/outputs/{experiment_name})
        
    Returns:
        Tuple of (train_losses, val_losses, eval_metrics)
    """
    # Load full YAML config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return _train_from_config_dict(config_dict, device, experiment_name, output_dir)


def _train_from_config_dict(config_dict: Dict[str, Any], device: Optional[str] = None, experiment_name: Optional[str] = None, output_dir: Optional[str] = None):
    """
    Internal function to train from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        device: Device to use (default: auto-detect)
        experiment_name: Override experiment name
        output_dir: Optional output directory (default: gmi_data/outputs/{experiment_name})
        
    Returns:
        Tuple of (train_losses, val_losses, eval_metrics)
    """
    # Extract components using the new naming convention
    train_dataset = config_dict.get('train_dataset')
    validation_dataset = config_dict.get('validation_dataset')
    test_dataset = config_dict.get('test_dataset')
    diffusion_backbone = config_dict.get('diffusion_backbone')
    
    # Load components if they are config dictionaries
    if isinstance(train_dataset, dict):
        train_dataset = load_object_from_dict(train_dataset)
    if isinstance(validation_dataset, dict):
        validation_dataset = load_object_from_dict(validation_dataset)
    if isinstance(test_dataset, dict):
        test_dataset = load_object_from_dict(test_dataset)
    if isinstance(diffusion_backbone, dict):
        diffusion_backbone = load_object_from_dict(diffusion_backbone)
    
    # Use train_dataset as the main dataset
    dataset = train_dataset
    
    # Validation and test datasets are already loaded above
    
    if not all([dataset, diffusion_backbone]):
        raise ValueError("Configuration must contain 'train_dataset' and 'diffusion_backbone'")
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create diffusion model
    diffusion_model = DiffusionModel(
        diffusion_backbone=diffusion_backbone
    )
    
    # Move to device
    diffusion_model = diffusion_model.to(device)
    
    # Get experiment name
    if experiment_name is None:
        experiment_name = config_dict.get('experiment_name', 'unnamed_diffusion_experiment')
    
    # Update config_dict with the final experiment name before saving
    config_dict['experiment_name'] = experiment_name
    
    print(f"Starting training for experiment: {experiment_name}")
    
    # Determine output directory
    if output_dir is None:
        experiment_dir = Path(f"gmi_data/outputs/{experiment_name}")
    else:
        experiment_dir = Path(output_dir) / experiment_name
    
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the final combined config to the experiment directory
    config_save_path = experiment_dir / "final_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Saved final configuration to: {config_save_path}")
    
    # Extract training configuration
    training_config = config_dict.get('training', {})

    # Explicitly cast numeric values to correct types
    def to_int(val, default):
        try:
            return int(val)
        except Exception:
            return int(default)
    def to_float(val, default):
        try:
            return float(val)
        except Exception:
            return float(default)

    # Cast all relevant fields
    num_epochs = to_int(training_config.get('num_epochs', 100), 100)
    num_iterations_train = to_int(training_config.get('num_iterations_train', 100), 100)
    learning_rate = to_float(training_config.get('learning_rate', 0.001), 0.001)
    batch_size = to_int(training_config.get('batch_size', 4), 4)
    num_workers = to_int(training_config.get('num_workers', 4), 4)
    patience = to_int(training_config.get('patience', 10), 10)
    val_loss_smoothing = to_float(training_config.get('val_loss_smoothing', 0.9), 0.9)
    min_delta = to_float(training_config.get('min_delta', 1e-6), 1e-6)
    num_iterations_val = training_config.get('num_iterations_val', 10)
    if num_iterations_val is not None and num_iterations_val != 'all':
        num_iterations_val = to_int(num_iterations_val, 10)
    num_iterations_test = training_config.get('num_iterations_test', 10)
    if num_iterations_test is not None and num_iterations_test != 'all':
        num_iterations_test = to_int(num_iterations_test, 10)
    test_plot_vmin = to_float(training_config.get('test_plot_vmin', 0), 0)
    test_plot_vmax = to_float(training_config.get('test_plot_vmax', 1), 1)
    ema_decay = to_float(training_config.get('ema_decay', 0.999), 0.999)
    
    # Reverse process parameters
    reverse_t_start = to_float(training_config.get('reverse_t_start', 1.0), 1.0)
    reverse_t_end = to_float(training_config.get('reverse_t_end', 0.0), 0.0)
    reverse_spacing = training_config.get('reverse_spacing', 'linear')
    reverse_sampler = training_config.get('reverse_sampler', 'euler')
    reverse_timesteps = to_int(training_config.get('reverse_timesteps', 50), 50)

    # Train the model with training config using updated parameter names
    train_losses, val_losses, eval_metrics = diffusion_model.train_diffusion_model(
        val_data=validation_dataset,
        test_data=test_dataset,
        experiment_name=experiment_name,
        output_dir=str(experiment_dir),
        num_epochs=num_epochs,
        num_iterations_train=num_iterations_train,
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        test_batch_size=batch_size,
        train_num_workers=num_workers,
        val_num_workers=num_workers,
        test_num_workers=num_workers,
        shuffle_train=training_config.get('shuffle_train', True),
        shuffle_val=training_config.get('shuffle_val', True),
        shuffle_test=training_config.get('shuffle_test', False),
        use_ema=training_config.get('use_ema', True),
        ema_decay=ema_decay,
        early_stopping=training_config.get('early_stopping', True),
        patience=patience,
        val_loss_smoothing=val_loss_smoothing,
        min_delta=min_delta,
        num_iterations_val=num_iterations_val,
        num_iterations_test=num_iterations_test,
        verbose=training_config.get('verbose', True),
        very_verbose=training_config.get('very_verbose', False),
        wandb_project=training_config.get('wandb_project', None),
        wandb_config=training_config.get('wandb_config', None),
        save_checkpoints=training_config.get('save_checkpoints', True),
        test_plot_vmin=test_plot_vmin,
        test_plot_vmax=test_plot_vmax,
        test_save_plots=training_config.get('test_save_plots', True),
        reverse_t_start=reverse_t_start,
        reverse_t_end=reverse_t_end,
        reverse_spacing=reverse_spacing,
        reverse_sampler=reverse_sampler,
        reverse_timesteps=reverse_timesteps
    )
    
    print(f"Training completed!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    return train_losses, val_losses, eval_metrics 