import torch
from torch import nn
try:  # optional training-only dependency (only used in the EMA training path)
    from torch_ema import ExponentialMovingAverage
except ImportError:  # keep the area importable without torch_ema installed
    ExponentialMovingAverage = None
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

from ..samplers import Sampler, DatasetSampler, DataLoaderSampler, ModuleSampler
from ..datasets.core import GMI_Dataset
from ..train import train
import torch.nn.functional as F

class ImageReconstructionTask(nn.Module):
    """
    Image Reconstruction Task for training image reconstruction models.
    
    SHAPE CONVENTIONS:
    - Images: (batch_size, channels, height, width) - e.g., (32, 1, 28, 28) for grayscale MNIST
    - Measurements: (batch_size, channels, height, width) - same spatial dimensions as images
    - Reconstructions: (batch_size, channels, height, width) - same spatial dimensions as images
    
    NOTE: When called from gmi.train(), batch_data will have shape (batch_size, channels, height, width)
    with only ONE batch dimension. The internal samplers may add extra batch dimensions that need
    to be handled appropriately.
    """
    def __init__(self, 
                 image_dataset,
                 measurement_simulator,
                 image_reconstructor,
                 device=None):
        
        # initialize the parent class
        super(ImageReconstructionTask, self).__init__()
        
        # if image_dataset is a torch.utils.data.Dataset, convert it to a DatasetSampler
        if isinstance(image_dataset, torch.utils.data.Dataset):
            image_dataset = DatasetSampler(image_dataset)

        # if image_dataset is a torch.utils.data.DataLoader, convert it to a DataLoaderSampler
        if isinstance(image_dataset, torch.utils.data.DataLoader):
            image_dataset = DataLoaderSampler(image_dataset)

        # if measurement_simulator is a torch.nn.Module, convert it to a ModuleSampler
        if isinstance(measurement_simulator, torch.nn.Module):
            measurement_simulator = ModuleSampler(measurement_simulator)

        # if image_reconstructor is a torch.nn.Module, convert it to a ModuleSampler
        if isinstance(image_reconstructor, torch.nn.Module):
            image_reconstructor = ModuleSampler(image_reconstructor)

        # assert that image_dataset, measurement_simulator, and image_reconstructor are instances of Sampler
        assert isinstance(image_dataset, Sampler), 'image_dataset must be an instance of torch.utils.data.Dataset, torch.utils.data.DataLoader, gmi.Sampler, or gmi.GMI_Dataset'
        assert isinstance(measurement_simulator, Sampler), 'measurement_simulator must be an instance of torch.nn.Module or gmi.Sampler'
        assert isinstance(image_reconstructor, Sampler), 'image_reconstructor must be an instance of torch.nn.Module or gmi.Sampler'

        # set the image_dataset, measurement_simulator, and image_reconstructor
        self.image_dataset = image_dataset
        self.measurement_simulator = measurement_simulator
        self.image_reconstructor = image_reconstructor
        
        self.device=device

        # Move models to device if possible
        if self.device is not None:
            # Move image_reconstructor's underlying module to device
            if hasattr(self.image_reconstructor, 'module') and hasattr(self.image_reconstructor.module, 'to'):
                self.image_reconstructor.module = self.image_reconstructor.module.to(self.device)
            elif hasattr(self.image_reconstructor, 'to'):
                self.image_reconstructor = self.image_reconstructor.to(self.device)
            # Move measurement_simulator's underlying module to device
            if hasattr(self.measurement_simulator, 'module') and hasattr(self.measurement_simulator.module, 'to'):
                self.measurement_simulator.module = self.measurement_simulator.module.to(self.device)
            elif hasattr(self.measurement_simulator, 'to'):
                self.measurement_simulator = self.measurement_simulator.to(self.device)

    @classmethod
    def from_config(cls, config_path: Union[str, Path], device=None):
        """
        Create an ImageReconstructionTask from a YAML configuration file.
        Args:
            config_path: Path to the YAML configuration file
            device: Device to place the models on
        Returns:
            ImageReconstructionTask instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config, device)
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any], device=None):
        """
        Create an ImageReconstructionTask from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            device: Device to place the models on
            
        Returns:
            ImageReconstructionTask instance
        """
        # Validate required components
        required_components = ['dataset', 'measurement_simulator', 'image_reconstructor']
        for component in required_components:
            if component not in config:
                raise ValueError(f"Missing required component '{component}' in configuration")
        
        # Load dataset
        dataset = cls._load_component(config['dataset'], 'dataset')
        
        # Load measurement simulator (must be a conditional random variable)
        measurement_simulator = cls._load_component(config['measurement_simulator'], 'measurement_simulator')
        
        # Validate that measurement_simulator is a conditional random variable
        from ..random_variable_gmi.gaussian import ConditionalGaussianRandomVariable
        if not isinstance(measurement_simulator, ConditionalGaussianRandomVariable):
            raise ValueError(f"measurement_simulator must be a subclass of ConditionalGaussianRandomVariable, got {type(measurement_simulator)}")
        
        # Load image reconstructor
        image_reconstructor = cls._load_component(config['image_reconstructor'], 'image_reconstructor')
        
        return cls(
            image_dataset=dataset,
            measurement_simulator=measurement_simulator,
            image_reconstructor=image_reconstructor,
            device=device
        )
    
    @staticmethod
    def _load_component(component_config: Dict[str, Any], component_name: str):
        """
        Load a component from configuration.
        
        Args:
            component_config: Component configuration dictionary
            component_name: Name of the component for error messages
            
        Returns:
            Loaded component instance
        """
        from ..config import load_object_from_dict
        return load_object_from_dict(component_config)

    class TestClosure(nn.Module):
        """
        Test closure for image reconstruction evaluation.
        Returns a dict with metrics and image file paths for WandB logging.
        """
        def __init__(self, parent, plot_vmin=0, plot_vmax=1, plot_save_dir=None, save_plots=False):
            super().__init__()
            self.parent = parent
            self.plot_vmin = plot_vmin
            self.plot_vmax = plot_vmax
            self.plot_save_dir = plot_save_dir
            self.save_plots = save_plots
        
        def _compute_rmse(self, pred, target):
            return torch.sqrt(F.mse_loss(pred, target)).item()
        def _compute_psnr(self, pred, target):
            mse = F.mse_loss(pred, target)
            if mse == 0:
                return float('inf')
            max_val = 1.0
            return 20 * torch.log10(max_val / torch.sqrt(mse)).item()
        def _compute_ssim(self, pred, target):
            mu1 = pred.mean()
            mu2 = target.mean()
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = pred.var()
            sigma2_sq = target.var()
            sigma12 = ((pred - mu1) * (target - mu2)).mean()
            c1 = 0.01 ** 2
            c2 = 0.03 ** 2
            ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            return ssim.item()
        def _compute_lpips(self, pred, target):
            return F.mse_loss(pred, target).item()
        
        def _save_reconstruction_plot(self, images, measurements, reconstructions, epoch, iteration):
            """Save a plot showing original image, measurement, and reconstruction (single example)."""
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path
            
            # Convert tensors to numpy for plotting
            images_np = images.cpu().detach().numpy()
            measurements_np = measurements.cpu().detach().numpy()
            reconstructions_np = reconstructions.cpu().detach().numpy()
            
            # Get number of channels
            channels = images_np.shape[1]
            
            # Create a simple 1-row, 3-column plot showing just the first example
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            
            # Original image (first sample)
            if channels == 1:
                axes[0].imshow(images_np[0, 0], cmap='gray', vmin=self.plot_vmin, vmax=self.plot_vmax)
            else:
                axes[0].imshow(np.transpose(images_np[0], (1, 2, 0)))
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Measurement (first sample)
            if channels == 1:
                axes[1].imshow(measurements_np[0, 0], cmap='gray', vmin=self.plot_vmin, vmax=self.plot_vmax)
            else:
                axes[1].imshow(np.transpose(measurements_np[0], (1, 2, 0)))
            axes[1].set_title('Measurement')
            axes[1].axis('off')
            
            # Reconstruction (first sample)
            if channels == 1:
                axes[2].imshow(reconstructions_np[0, 0], cmap='gray', vmin=self.plot_vmin, vmax=self.plot_vmax)
            else:
                axes[2].imshow(np.transpose(reconstructions_np[0], (1, 2, 0)))
            axes[2].set_title('Reconstruction')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            # Save the plot if requested
            plot_path = None
            if self.save_plots and self.plot_save_dir is not None:
                plot_dir = Path(self.plot_save_dir)
                plot_dir.mkdir(parents=True, exist_ok=True)
                plot_path = plot_dir / f"reconstruction_epoch_{epoch}_iter_{iteration}.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            
            plt.close()
            return str(plot_path) if plot_path else None
        
        def forward(self, batch_data, epoch=None, iteration=None):
            images = batch_data
            measurements = self.parent.sample_measurements_given_images(1, images)
            if measurements.dim() > images.dim():
                measurements = measurements[0]
            reconstructions = self.parent.sample_reconstructions_given_measurements(1, measurements)
            if reconstructions.dim() > images.dim():
                reconstructions = reconstructions[0]
            assert measurements.shape == images.shape
            assert reconstructions.shape == images.shape
            
            # Compute metrics
            rmse = self._compute_rmse(reconstructions, images)
            psnr = self._compute_psnr(reconstructions, images)
            ssim = self._compute_ssim(reconstructions, images)
            lpips = self._compute_lpips(reconstructions, images)
            
            # Generate and save reconstruction plot
            plot_path = self._save_reconstruction_plot(images, measurements, reconstructions, epoch, iteration)
            
            result = {
                'rmse': rmse,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips
            }
            
            # Add plot path to result if available
            if plot_path:
                result['reconstruction_plot'] = plot_path
            
            return result

    def train_image_reconstructor(self, 
                                 train_data=None,
                                 val_data=None,
                                 test_data=None,
                                 train_batch_size=4,
                                 val_batch_size=4,
                                 test_batch_size=4,
                                 train_num_workers=4,
                                 val_num_workers=4,
                                 test_num_workers=4,
                                 shuffle_train=True,
                                 shuffle_val=True,
                                 shuffle_test=False,
                                 num_epochs=100,
                                 num_iterations_train=100,
                                 num_iterations_val=10,
                                 num_iterations_test=1,
                                 learning_rate=1e-3,
                                 use_ema=True,
                                 ema_decay=0.999,
                                 early_stopping=True,
                                 patience=10,
                                 val_loss_smoothing=0.9,
                                 min_delta=1e-6,
                                 verbose=True,
                                 very_verbose=False,
                                 wandb_project=None,
                                 wandb_config=None,
                                 save_checkpoints=True,
                                 experiment_name=None,
                                 output_dir=None,
                                 epochs_per_evaluation=None,
                                 train_loss_closure=None,
                                 val_loss_closure=None,
                                 test_closure=None,
                                 test_plot_vmin=0,
                                 test_plot_vmax=1,
                                 test_save_plots=True,
                                 final_test_iterations='all',
                                 **kwargs):
        """
        Train the image reconstructor using the gmi.train function.
        
        Args:
            train_data: Training dataset (PyTorch Dataset). If None, uses self.image_dataset
            val_data: Validation dataset (PyTorch Dataset). If None, no validation is performed
            test_data: Test dataset (PyTorch Dataset). Used for evaluation metrics
            train_batch_size: Batch size for training (default: 4)
            val_batch_size: Batch size for validation (default: 4)
            test_batch_size: Batch size for testing (default: 4)
            train_num_workers: Number of workers for training DataLoader (default: 4)
            val_num_workers: Number of workers for validation DataLoader (default: 4)
            test_num_workers: Number of workers for test DataLoader (default: 4)
            shuffle_train: Whether to shuffle training data (default: True)
            shuffle_val: Whether to shuffle validation data (default: True)
            shuffle_test: Whether to shuffle test data (default: False)
            num_epochs: Number of training epochs
            num_iterations_train: Number of training iterations per epoch
            num_iterations_val: Number of validation iterations per epoch
            num_iterations_test: Number of test iterations per epoch
            learning_rate: Learning rate for the optimizer
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay factor
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
            val_loss_smoothing: Validation loss smoothing factor
            min_delta: Minimum change for early stopping
            verbose: Whether to print training progress
            very_verbose: Whether to print very detailed progress
            wandb_project: WandB project name
            wandb_config: WandB configuration
            save_checkpoints: Whether to save model checkpoints
            experiment_name: Name for the experiment (used for saving outputs)
            output_dir: Output directory for saving model checkpoints and logs
            epochs_per_evaluation: Run evaluation every N epochs
            train_loss_closure: Custom training loss closure (default: MSE loss)
            val_loss_closure: Custom validation loss closure (default: same as train)
            test_closure: Custom test closure (default: TestClosure with metrics and plots)
            test_plot_vmin: Minimum value for test plot colorbar (default: 0)
            test_plot_vmax: Maximum value for test plot colorbar (default: 1)
            test_save_plots: Whether to save reconstruction plots during testing (default: True)
            final_test_iterations: Number of iterations for final test evaluation. If set to 'all', will run over the full test dataset sequentially and save per-sample metrics (rmse, psnr, ssim, lpips) to a CSV for histogram analysis.
            **kwargs: Additional arguments passed to gmi.train
        Returns:
            Tuple of (train_losses, val_losses, eval_metrics)
        """
        # Helper to extract underlying dataset if DatasetSampler is passed
        def get_dataset(data):
            if hasattr(data, 'dataset'):
                return data.dataset
            return data

        # Use self.image_dataset if no train_data is provided
        if train_data is None:
            train_data = get_dataset(self.image_dataset)
        else:
            train_data = get_dataset(train_data)
        if val_data is not None:
            val_data = get_dataset(val_data)
        if test_data is not None:
            test_data = get_dataset(test_data)

        # Set device
        device = self.device if self.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create default closures if not provided
        if train_loss_closure is None:
            def mse_loss(pred, target):
                return F.mse_loss(pred, target)
            train_loss_closure = self.loss_closure(mse_loss)
        
        if val_loss_closure is None:
            val_loss_closure = train_loss_closure
        
        if test_closure is None and test_data is not None:
            # Create test closure with image saving enabled for WandB logging
            test_plot_save_dir = None
            if experiment_name:
                # Create output directory for plots
                output_dir = Path(output_dir) if output_dir else Path(f"gmi_data/outputs/{experiment_name}")
                output_dir.mkdir(parents=True, exist_ok=True)
                test_plot_save_dir = output_dir / "test_plots"
                test_plot_save_dir.mkdir(exist_ok=True)
            
            test_closure = self.TestClosure(
                parent=self,
                plot_vmin=test_plot_vmin,
                plot_vmax=test_plot_vmax,
                plot_save_dir=str(test_plot_save_dir) if test_plot_save_dir else None,
                save_plots=test_save_plots
            )
        
        # Setup model saving if requested
        save_best_model_path = None
        model_to_save = None
        if save_checkpoints and experiment_name:
            # Create output directory
            output_dir = Path(output_dir) if output_dir else Path(f"gmi_data/outputs/{experiment_name}")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_best_model_path = output_dir / "best_model.pth"
            model_to_save = self.image_reconstructor
        
        # Run training with new interface
        train_losses, val_losses, eval_metrics = train(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            train_num_workers=train_num_workers,
            val_num_workers=val_num_workers,
            test_num_workers=test_num_workers,
            shuffle_train=shuffle_train,
            shuffle_val=shuffle_val,
            shuffle_test=shuffle_test,
            train_loss_closure=train_loss_closure,
            val_loss_closure=val_loss_closure,
            test_closure=test_closure,
            num_epochs=num_epochs,
            num_iterations=num_iterations_train,
            num_iterations_val=num_iterations_val,
            num_iterations_test=num_iterations_test,
            final_test_iterations=final_test_iterations,
            lr=learning_rate,
            use_ema=use_ema,
            ema_decay=ema_decay,
            early_stopping=early_stopping,
            patience=patience,
            val_loss_smoothing=val_loss_smoothing,
            min_delta=min_delta,
            verbose=verbose,
            very_verbose=very_verbose,
            wandb_project=wandb_project,
            wandb_config=wandb_config,
            save_best_model_path=save_best_model_path,
            model_to_save=model_to_save,
            device=device,
            experiment_name=experiment_name,
            **kwargs
        )
        
        # Download WandB data for this specific run if WandB was used
        if wandb_project is not None:
            try:
                self._download_wandb_run_data(experiment_name, wandb_project, save_best_model_path)
            except Exception as e:
                print(f"Warning: Failed to download WandB data: {e}")
        
        # Run final evaluation if test_data is provided and final_test_iterations is set
        if test_data is not None and final_test_iterations is not None:
            print(f"\nRunning final evaluation with {final_test_iterations} iterations...")
            final_eval_output_dir = None
            if save_best_model_path is not None:
                final_eval_output_dir = Path(save_best_model_path).parent
            elif output_dir is not None:
                final_eval_output_dir = Path(output_dir)
            
            try:
                final_eval_summary = self.run_final_evaluation(
                    test_data=test_data,
                    output_dir=str(final_eval_output_dir) if final_eval_output_dir else None,
                    experiment_name=experiment_name,
                    verbose=verbose
                )
                print("Final evaluation completed successfully!")
            except Exception as e:
                print(f"Warning: Final evaluation failed: {e}")
        
        return train_losses, val_losses, eval_metrics

    def run_final_evaluation(self, test_data, output_dir=None, experiment_name=None, verbose=True):
        """
        Run final evaluation on the full test dataset and save per-sample metrics.
        
        Args:
            test_data: Test dataset to evaluate on
            output_dir: Output directory for saving results
            experiment_name: Experiment name for file naming
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing summary statistics
        """
        import numpy as np
        import csv
        from pathlib import Path
        from tqdm import tqdm
        import torch.utils.data
        
        # Set device
        device = self.device if self.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create DataLoader if needed
        if isinstance(test_data, torch.utils.data.Dataset):
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        elif isinstance(test_data, torch.utils.data.DataLoader):
            test_loader = test_data
        else:
            raise ValueError("test_data must be a PyTorch Dataset or DataLoader")
        
        print(f"\nRunning final evaluation on {len(test_loader)} samples...")
        
        per_sample_metrics = []
        
        # Put models in eval mode
        if hasattr(self.image_reconstructor, 'module') and hasattr(self.image_reconstructor.module, 'eval'):
            self.image_reconstructor.module.eval()
        elif hasattr(self.image_reconstructor, 'eval'):
            self.image_reconstructor.eval()
            
        if hasattr(self.measurement_simulator, 'module') and hasattr(self.measurement_simulator.module, 'eval'):
            self.measurement_simulator.module.eval()
        elif hasattr(self.measurement_simulator, 'eval'):
            self.measurement_simulator.eval()
        
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(test_loader, desc="Final Evaluation")):
                # Handle different data formats
                if isinstance(batch_data, (tuple, list)):
                    images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                elif isinstance(batch_data, torch.Tensor):
                    images = batch_data.to(device)
                else:
                    continue
                
                # Create measurements and reconstructions
                measurements = self.sample_measurements_given_images(1, images)
                if measurements.dim() > images.dim():
                    measurements = measurements[0]
                    
                reconstructions = self.sample_reconstructions_given_measurements(1, measurements)
                if reconstructions.dim() > images.dim():
                    reconstructions = reconstructions[0]
                
                # Compute metrics for each sample in the batch
                for j in range(images.shape[0]):
                    img = images[j:j+1]
                    rec = reconstructions[j:j+1]
                    
                    # Compute metrics using the same methods as TestClosure
                    rmse = self._compute_rmse(rec, img)
                    psnr = self._compute_psnr(rec, img)
                    ssim = self._compute_ssim(rec, img)
                    lpips = self._compute_lpips(rec, img)
                    
                    per_sample_metrics.append({
                        'index': len(per_sample_metrics),
                        'rmse': rmse,
                        'psnr': psnr,
                        'ssim': ssim,
                        'lpips': lpips
                    })
        
        # Save per-sample metrics to CSV
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            per_sample_csv = output_path / "final_evaluation_metrics_per_sample.csv"
            with open(per_sample_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['index', 'rmse', 'psnr', 'ssim', 'lpips'])
                writer.writeheader()
                for row in per_sample_metrics:
                    writer.writerow(row)
            
            if verbose:
                print(f"Per-sample metrics saved to: {per_sample_csv}")
        
        # Compute summary statistics
        summary_stats = {}
        if per_sample_metrics:
            for metric in ['rmse', 'psnr', 'ssim', 'lpips']:
                values = np.array([row[metric] for row in per_sample_metrics])
                summary_stats[f'{metric}_mean'] = float(np.mean(values))
                summary_stats[f'{metric}_std'] = float(np.std(values))
                summary_stats[f'{metric}_min'] = float(np.min(values))
                summary_stats[f'{metric}_max'] = float(np.max(values))
        
        # Print summary if verbose
        if verbose and per_sample_metrics:
            print(f"\nFinal evaluation summary:")
            for metric in ['rmse', 'psnr', 'ssim', 'lpips']:
                values = [row[metric] for row in per_sample_metrics]
                print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}")
        
        return summary_stats

    def _compute_rmse(self, pred, target):
        """Compute Root Mean Square Error."""
        return torch.sqrt(F.mse_loss(pred, target)).item()

    def _create_evaluation_function(self, test_data, batch_size):
        """
        Create an evaluation function that computes PSNR, SSIM, and LPIPS on test data.
        
        Args:
            test_data: Test dataset to use for evaluation
            batch_size: Batch size for evaluation
            
        Returns:
            Evaluation function that takes (model, wandb_project, wandb_config, epoch) and returns metrics
        """
        def evaluation_function(model, wandb_project, wandb_config, epoch):
            """
            Evaluation function that computes PSNR, SSIM, and LPIPS on test data.
            
            Args:
                model: The trained model
                wandb_project: WandB project name
                wandb_config: WandB configuration
                epoch: Current epoch number
                
            Returns:
                Dictionary of evaluation metrics
            """
            # Use the provided test_data
            test_dataset = test_data
            
            # Compute metrics on 10 batches
            num_batches = 10
            psnr_values = []
            ssim_values = []
            lpips_values = []
            
            model.eval()
            with torch.no_grad():
                for i in range(num_batches):
                    # Sample test data - handle both Sampler and Dataset objects
                    if hasattr(test_dataset, 'sample'):
                        # If it's a Sampler object
                        test_images = test_dataset.sample(batch_size)
                    else:
                        # If it's a PyTorch Dataset, use random sampling
                        import random
                        indices = random.sample(range(len(test_dataset)), min(batch_size, len(test_dataset)))
                        test_images = torch.stack([test_dataset[idx] for idx in indices])
                    
                    if self.device is not None:
                        test_images = test_images.to(self.device)
                    
                    # Create noisy measurements
                    measurements = self.sample_measurements_given_images(batch_size, test_images)
                    
                    # Reconstruct images
                    reconstructions = self.sample_reconstructions_given_measurements(batch_size, measurements)
                    
                    # Compute metrics
                    psnr = self._compute_psnr(reconstructions, test_images)
                    ssim = self._compute_ssim(reconstructions, test_images)
                    lpips = self._compute_lpips(reconstructions, test_images)
                    
                    psnr_values.append(psnr)
                    ssim_values.append(ssim)
                    lpips_values.append(lpips)
            
            # Average metrics
            avg_psnr = sum(psnr_values) / len(psnr_values)
            avg_ssim = sum(ssim_values) / len(ssim_values)
            avg_lpips = sum(lpips_values) / len(lpips_values)
            
            metrics = {
                'test_psnr': avg_psnr,
                'test_ssim': avg_ssim,
                'test_lpips': avg_lpips
            }
            
            # Log to WandB if available
            if wandb_project is not None:
                import wandb
                wandb.log({
                    'test_psnr': avg_psnr,
                    'test_ssim': avg_ssim,
                    'test_lpips': avg_lpips,
                    'epoch': epoch
                })
            
            return metrics
        
        return evaluation_function

    def _compute_psnr(self, pred, target):
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        max_val = 1.0  # Assuming normalized images
        return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

    def _compute_ssim(self, pred, target):
        """Compute Structural Similarity Index."""
        # Simple SSIM implementation - could be replaced with more sophisticated version
        mu1 = pred.mean()
        mu2 = target.mean()
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = pred.var()
        sigma2_sq = target.var()
        sigma12 = ((pred - mu1) * (target - mu2)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        return ssim.item()

    def _compute_lpips(self, pred, target):
        """Compute Learned Perceptual Image Patch Similarity."""
        # Simple LPIPS approximation using MSE in feature space
        # In practice, you'd want to use a proper LPIPS implementation
        return F.mse_loss(pred, target).item()

    @classmethod
    def run_from_yaml(cls, config_path: Union[str, Path], operations: Optional[List[str]] = None, device=None):
        """
        Run operations specified in a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
            operations: List of operations to run. If None, runs all operations found in the config.
                       Supported operations: 'init', 'train', 'sample', 'evaluate'
            device: Device to use for the task
            
        Returns:
            ImageReconstructionTask instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"YAML configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # If no operations specified, run all available operations
        if operations is None:
            operations = []
            if 'experiment_name' in config:
                operations.append('init')
            if 'training' in config:
                operations.append('train')
            if 'sampling' in config:
                operations.append('sample')
            if 'evaluation' in config:
                operations.append('evaluate')
        
        task = None
        
        for operation in operations:
            print(f"Running operation: {operation}")
            
            if operation == 'init':
                # Initialize the task from config
                task = cls.from_dict(config, device=device)
                print(f"Initialized ImageReconstructionTask with experiment: {config.get('experiment_name', 'unnamed')}")
                
            elif operation == 'train':
                # Train the image reconstructor
                if task is None:
                    # Initialize task if not already done
                    task = cls.from_dict(config, device=device)
                
                if 'training' not in config:
                    raise ValueError("Training configuration not found in YAML file")
                
                training_config = config['training']
                experiment_name = config.get('experiment_name', 'unnamed_experiment')
                
                # Extract training parameters
                train_params = {
                    'num_epochs': training_config.get('num_epochs', 100),
                    'num_iterations': training_config.get('num_iterations', 100),
                    'learning_rate': training_config.get('learning_rate', 0.001),
                    'train_batch_size': training_config.get('batch_size', 4),
                    'val_batch_size': training_config.get('batch_size', 4),
                    'test_batch_size': training_config.get('batch_size', 4),
                    'train_num_workers': training_config.get('num_workers', 4),
                    'val_num_workers': training_config.get('num_workers', 4),
                    'test_num_workers': training_config.get('num_workers', 4),
                    'shuffle_train': training_config.get('shuffle_train', True),
                    'shuffle_val': training_config.get('shuffle_val', True),
                    'shuffle_test': training_config.get('shuffle_test', False),
                    'use_ema': training_config.get('use_ema', True),
                    'ema_decay': training_config.get('ema_decay', 0.999),
                    'early_stopping': training_config.get('early_stopping', True),
                    'patience': training_config.get('patience', 10),
                    'val_loss_smoothing': training_config.get('val_loss_smoothing', 0.9),
                    'min_delta': training_config.get('min_delta', 1e-6),
                    'num_iterations_val': training_config.get('num_iterations_val', None),
                    'num_iterations_test': training_config.get('num_iterations_test', None),
                    'final_test_iterations': training_config.get('final_test_iterations', None),
                    'verbose': training_config.get('verbose', True),
                    'very_verbose': training_config.get('very_verbose', False),
                    'wandb_project': training_config.get('wandb_project', None),
                    'wandb_config': training_config.get('wandb_config', None),
                    'save_checkpoints': training_config.get('save_checkpoints', True),
                    'experiment_name': experiment_name,
                    'eval_fn': training_config.get('eval_fn', None),
                    'epochs_per_evaluation': training_config.get('epochs_per_evaluation', None),
                }
                
                # Run training
                train_losses, val_losses, eval_metrics = task.train_image_reconstructor(**train_params)
                print(f"Training completed. Final train loss: {train_losses[-1]:.4f}")
                if val_losses:
                    print(f"Final validation loss: {val_losses[-1]:.4f}")
                
            elif operation == 'sample':
                # Generate samples
                if task is None:
                    task = cls.from_dict(config, device=device)
                
                if 'sampling' not in config:
                    raise ValueError("Sampling configuration not found in YAML file")
                
                sampling_config = config['sampling']
                # TODO: Implement sampling functionality
                print("Sampling operation not yet implemented")
                
            elif operation == 'evaluate':
                # Evaluate the model
                if task is None:
                    task = cls.from_dict(config, device=device)
                
                if 'evaluation' not in config:
                    raise ValueError("Evaluation configuration not found in YAML file")
                
                evaluation_config = config['evaluation']
                # TODO: Implement evaluation functionality
                print("Evaluation operation not yet implemented")
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        return task

    def sample_images(self, image_batch_size):
        """
        Sample images from the image dataset.
        
        Args:
            image_batch_size: Number of images to sample
            
        Returns:
            images: Shape (image_batch_size, channels, height, width)
                   e.g., (32, 1, 28, 28) for grayscale MNIST
        """
        images = self.image_dataset.sample(image_batch_size)
        if self.device is not None:
            images = images.to(self.device)
        return images
    
    def sample_measurements_given_images(self, measurement_batch_size, images):
        """
        Sample measurements given images using the measurement simulator.
        
        Args:
            measurement_batch_size: Number of measurements to generate per image
            images: Shape (image_batch_size, channels, height, width)
                   e.g., (32, 1, 28, 28) for grayscale MNIST
            
        Returns:
            measurements: Shape depends on sampler implementation:
                         - If measurement_batch_size=1: (image_batch_size, channels, height, width)
                         - If measurement_batch_size>1: (measurement_batch_size, image_batch_size, channels, height, width)
                         
            NOTE: The sampler may add extra batch dimensions. When called from gmi.train(),
                  we expect measurement_batch_size=1 and want shape (image_batch_size, channels, height, width)
        """
        measurements = self.measurement_simulator.sample(measurement_batch_size, images)
        if self.device is not None:
            measurements = measurements.to(self.device)
        return measurements
    
    def sample_reconstructions_given_measurements(self, reconstruction_batch_size, measurements):
        """
        Sample reconstructions given measurements using the image reconstructor.
        
        Args:
            reconstruction_batch_size: Number of reconstructions to generate per measurement
            measurements: Shape depends on previous step:
                         - If from single measurement: (image_batch_size, channels, height, width)
                         - If from multiple measurements: (measurement_batch_size, image_batch_size, channels, height, width)
            
        Returns:
            reconstructions: Shape depends on sampler implementation:
                            - If reconstruction_batch_size=1: same as measurements input
                            - If reconstruction_batch_size>1: (reconstruction_batch_size, ..., channels, height, width)
                            
            NOTE: The sampler may add extra batch dimensions. When called from gmi.train(),
                  we expect reconstruction_batch_size=1 and want shape (image_batch_size, channels, height, width)
        """
        reconstructions = self.image_reconstructor(reconstruction_batch_size, measurements)
        if self.device is not None:
            reconstructions = reconstructions.to(self.device)
        return reconstructions
    
    def sample_images_measurements(self, image_batch_size, measurement_batch_size):
        """
        Sample images and corresponding measurements.
        
        Args:
            image_batch_size: Number of images to sample
            measurement_batch_size: Number of measurements per image
            
        Returns:
            images: Shape (1, image_batch_size, channels, height, width) - extra batch dim added
            measurements: Shape (measurement_batch_size, image_batch_size, channels, height, width)
                         
        NOTE: This method adds extra batch dimensions for compatibility with the complex
              sampling pipeline. For simple training/testing, use the individual methods.
        """
        # call the image_dataset sampler
        images = self.sample_images(image_batch_size)  # Shape: (image_batch_size, channels, height, width)
        # call the measurement_simulator conditional sampler
        measurements = self.sample_measurements_given_images(measurement_batch_size, images)
        assert isinstance(measurements, torch.Tensor), 'sample_measurements_given_images() must return a torch.Tensor'
        # add a dimension for the measurement batches
        images = images.unsqueeze(0)  # Shape: (1, image_batch_size, channels, height, width)
        return images, measurements
    
    def sample_images_measurements_reconstructions(self, image_batch_size, measurement_batch_size, reconstruction_batch_size):
        """
        Sample images, measurements, and reconstructions with multiple batch dimensions.
        
        Args:
            image_batch_size: Number of images to sample
            measurement_batch_size: Number of measurements per image
            reconstruction_batch_size: Number of reconstructions per measurement
            
        Returns:
            images: Shape (1, 1, image_batch_size, channels, height, width)
            measurements: Shape (1, measurement_batch_size, image_batch_size, channels, height, width)
            reconstructions: Shape (reconstruction_batch_size, measurement_batch_size, image_batch_size, channels, height, width)
                            
        NOTE: This method creates a complex multi-dimensional batch structure for advanced
              sampling scenarios. For simple training/testing, use the individual methods.
        """
        # call the image_dataset sampler and the measurement_simulator conditional sampler
        images, measurements = self.sample_images_measurements(image_batch_size, measurement_batch_size)

        # get shapes
        image_shape = images.shape[2:]
        measurement_shape = measurements.shape[2:]

        # combine the image and measurement batch dimensions into one batch dimension
        measurements = measurements.view(image_batch_size*measurement_batch_size, *measurements.shape[2:])
        # call the image_reconstructor conditional sampler
        reconstructions = self.sample_reconstructions_given_measurements(reconstruction_batch_size, measurements)
        assert isinstance(reconstructions, torch.Tensor), 'sample_reconstructions_given_measurements() must return a torch.Tensor'
        # reshape the images to have reconstruction, measurement, and image batch dimensions
        images = images.view(1,1,image_batch_size, *image_shape)
        # reshape the measurements to have reconstruction, measurement, and image batch dimensions
        measurements = measurements.view(1, measurement_batch_size, image_batch_size, *measurement_shape)
        # reshape the reconstructions to have reconstruction, measurement, and image batch dimensions
        reconstructions = reconstructions.view(reconstruction_batch_size, measurement_batch_size, image_batch_size, *image_shape)
        return images, measurements, reconstructions

    def forward(self, image_batch_size, measurement_batch_size, reconstruction_batch_size):
        return self.sample_images_measurements_reconstructions(image_batch_size, measurement_batch_size, reconstruction_batch_size)

    def loss_closure(self, loss_fn):
        class LossClosure(nn.Module):
            def __init__(self, parent, model, loss_fn):
                super(LossClosure, self).__init__()
                self.parent = parent  # reference to the parent class
                self.model = model
                self.loss_fn = loss_fn

            def forward(self, batch_data):
                """
                Forward pass for training loss computation.
                
                Args:
                    batch_data: Batch of images with shape (batch_size, channels, height, width)
                               e.g., (32, 1, 28, 28) for grayscale MNIST
                
                Returns:
                    loss: Scalar loss value
                """
                images = batch_data  # Shape: (batch_size, channels, height, width)
                assert isinstance(self.parent, ImageReconstructionTask)
                
                # Create measurements: expect shape (batch_size, channels, height, width)
                # If sampler returns extra batch dimension, remove it
                measurements = self.parent.sample_measurements_given_images(1, images)
                if measurements.dim() > images.dim():
                    measurements = measurements[0]  # Remove extra batch dimension
                
                # Create reconstructions: expect shape (batch_size, channels, height, width)
                # If sampler returns extra batch dimension, remove it
                reconstructions = self.parent.sample_reconstructions_given_measurements(1, measurements)
                if reconstructions.dim() > images.dim():
                    reconstructions = reconstructions[0]  # Remove extra batch dimension
                
                # Verify shapes are consistent
                assert measurements.shape == images.shape, f"Measurement shape {measurements.shape} != image shape {images.shape}"
                assert reconstructions.shape == images.shape, f"Reconstruction shape {reconstructions.shape} != image shape {images.shape}"
                
                loss = self.loss_fn(reconstructions, images)
                return loss
                
        return LossClosure(self, self.image_reconstructor, loss_fn) 

    def _download_wandb_run_data(self, experiment_name, wandb_project, save_best_model_path):
        """
        Download WandB data for a specific run including GPU metrics.
        
        Args:
            experiment_name: Name of the experiment (used to find the run)
            wandb_project: WandB project name
            save_best_model_path: Path where model was saved (used for output directory)
        """
        try:
            import wandb
            import pandas as pd
            from pathlib import Path
            
            # Find the run by name
            api = wandb.Api()
            runs = api.runs(f"{wandb_project}")
            
            # Find the run with matching name
            target_run = None
            for run in runs:
                if run.name == experiment_name:
                    target_run = run
                    break
            
            if target_run is None:
                print(f"Could not find WandB run with name: {experiment_name}")
                return
            
            print(f"Found WandB run: {experiment_name} (ID: {target_run.id})")
            
            # Create output directory
            if save_best_model_path is not None:
                output_dir = Path(save_best_model_path).parent / "wandb_data"
            else:
                # Fallback to default location if no save path provided
                output_dir = Path(f"gmi_data/outputs/{experiment_name}/wandb_data")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Downloading WandB data to: {output_dir}")
            
            # Download regular run history (metrics, config, summary)
            print("  Downloading run history...")
            history = target_run.history()
            history_path = output_dir / "run_history.csv"
            history.to_csv(history_path, index=False)
            print(f"    Saved {len(history)} history records to {history_path}")
            
            # Download run config
            print("  Downloading run config...")
            config_path = output_dir / "run_config.json"
            with open(config_path, 'w') as f:
                import json
                json.dump(target_run.config, f, indent=2)
            print(f"    Saved config to {config_path}")
            
            # Download run summary
            print("  Downloading run summary...")
            summary_path = output_dir / "run_summary.json"
            with open(summary_path, 'w') as f:
                import json
                json.dump(target_run.summary, f, indent=2)
            print(f"    Saved summary to {summary_path}")
            
            # Download GPU and system metrics using official API
            print("  Downloading GPU and system metrics...")
            try:
                system_metrics = target_run.history(stream="systemMetrics")
                if not system_metrics.empty:
                    system_metrics_path = output_dir / "gpu_system_metrics.csv"
                    system_metrics.to_csv(system_metrics_path, index=False)
                    print(f"    Saved {len(system_metrics)} system metric records to {system_metrics_path}")
                    print(f"    GPU metrics columns: {[col for col in system_metrics.columns if 'gpu' in col.lower()]}")
                else:
                    print("    No system metrics found")
            except Exception as e:
                print(f"    Warning: Could not download system metrics: {e}")
            
            # Download files
            print("  Downloading files...")
            for file in target_run.files():
                try:
                    file_path = output_dir / file.name
                    file.download(root=str(output_dir))
                    print(f"    Downloaded file: {file.name}")
                except Exception as e:
                    print(f"    Error downloading file {file.name}: {e}")
            
            print("✅ WandB data download completed for", experiment_name)
            
        except Exception as e:
            print(f"Error downloading WandB data: {e}")
            import traceback
            traceback.print_exc()

# from ..diffusion import UnconditionalDiffusionModel

# class DiffusionBridgeImageReconstructor(nn.Module):
#             def __init__(self, initial_reconstructor, diffusion_model, final_reconstructor):
#                 super(DiffusionBridgeImageReconstructor, self).__init__()
#                 assert isinstance(initial_reconstructor, nn.Module)
#                 assert isinstance(diffusion_model, UnconditionalDiffusionModel)
#                 assert isinstance(final_reconstructor, nn.Module)
#                 self.initial_reconstructor = initial_reconstructor
#                 self.diffusion_model = diffusion_model
#                 self.final_reconstructor = final_reconstructor
                
#             def forward(self,measurements,timesteps=None, num_timesteps=None, sampler='euler', verbose=False):

#                 assert isinstance(self.initial_reconstructor, nn.Module)
#                 assert isinstance(self.diffusion_model, UnconditionalDiffusionModel)
#                 assert isinstance(self.final_reconstructor, nn.Module)

#                 x_1 = self.initial_reconstructor(measurements)

#                 if timesteps is None:
#                     if num_timesteps is None:
#                         num_timesteps = 32
#                     timesteps = torch.linspace(1, 0, num_timesteps+1).to(x_1.device)
              
#                 assert isinstance(timesteps, torch.Tensor)
#                 assert timesteps.ndim == 1
#                 assert timesteps.shape[0] > 1
#                 assert timesteps[0] == 1.0
#                 assert timesteps[-1] == 0.0
#                 for i in range(1, timesteps.shape[0]):
#                     assert timesteps[i] < timesteps[i-1]

#                 x_0 = self.diffusion_model.reverse_sample(x_1, timesteps, sampler=sampler, return_all=False, verbose=verbose)
                
#                 reconstructions = self.final_reconstructor(x_0)
#                 return reconstructions

# class DiffusionBridgeModel(ImageReconstructionTask):
#     def __init__(self, 
#                  image_dataset,
#                  measurement_simulator,
#                  image_reconstructor,                 
#                  task_evaluator='rmse'):
        
#         assert isinstance(image_reconstructor, DiffusionBridgeImageReconstructor)
            
#         super(DiffusionBridgeModel, self).__init__(image_dataset, 
#                                                     measurement_simulator, 
#                                                     image_reconstructor, 
#                                                     task_evaluator=task_evaluator)
        
#     def train_diffusion_backbone(self, 
#                                  *args, 
#                                  num_epochs=100, 
#                                  num_iterations_per_epoch=100, 
#                                  num_epochs_per_save=None, 
#                                  weights_filename=None,
#                                  optimizer=None, 
#                                  verbose=True, 
#                                  time_sampler=None,
#                                  ema=False,
#                                  **kwargs):
                
#         assert isinstance(self.image_reconstructor, DiffusionBridgeImageReconstructor)

#         if optimizer is None:
#             optimizer = torch.optim.Adam(self.image_reconstructor.diffusion_model.diffusion_backbone.parameters(), lr=1e-3)

#         if ema:
#             ema =  ExponentialMovingAverage(self.image_reconstructor.diffusion_model.diffusion_backbone.parameters(), decay=0.995)
        
#         if time_sampler is None:
#             assert isinstance(self.image_reconstructor.diffusion_model.diffusion_backbone, torch.nn.Module)
#             time_sampler = lambda x_shape: torch.rand(x_shape)

#         if num_epochs_per_save is not None:
#             assert weights_filename is not None
#             assert isinstance(weights_filename, str)

#         train_loss = torch.zeros(num_epochs, dtype=torch.float32)

#         for epoch in range(num_epochs):
#             loss_sum = 0
#             for iteration in range(num_iterations_per_epoch):
#                 optimizer.zero_grad()
#                 x_0 = self.sample_images(*args, **kwargs)
#                 batch_size = x_0.shape[0]
#                 t = time_sampler((batch_size, 1)).to(x_0.device)
#                 noise = torch.randn_like(x_0)
#                 x_t = self.image_reconstructor.diffusion_model.sample_x_t_given_x_0_and_noise(x_0, noise, t) # forward process
                
                
#                 # x_0_pred = self.image_reconstructor.diffusion_model.predict_x_0_given_x_t(x_t, t) # reverse prediction
#                 # loss = torch.mean((x_0_pred - x_0)**2.0)

#                 noise_pred = self.image_reconstructor.diffusion_model.predict_noise_given_x_t(x_t, t) # reverse prediction
#                 loss = torch.mean((noise_pred - noise)**2.0)

#                 soft_tissue_mask = x_0 < 1.5
#                 loss += 9.0*torch.mean((noise_pred[soft_tissue_mask] - noise[soft_tissue_mask])**2.0)


#                 loss.backward()
#                 optimizer.step()
#                 loss_sum += loss.item()

#                 if ema:
#                     ema.update()

#             if num_epochs_per_save is not None and epoch % num_epochs_per_save == 0:
#                 torch.save(self.image_reconstructor.diffusion_model.diffusion_backbone.state_dict(), weights_filename)

            
#             train_loss[epoch] = loss_sum/num_iterations_per_epoch
            
#             if verbose:
#                 print(f"Epoch {epoch}: Loss: {train_loss[epoch].item()}")

#         return train_loss
    




