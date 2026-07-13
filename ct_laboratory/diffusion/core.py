import torch
from torch import nn
from ..sde import LinearSDE, StandardWienerSDE
from ..random_variable_gmi import UniformRandomVariable
from ..samplers import Sampler
from ..linear_system import InvertibleLinearSystem


class DiffusionModel(torch.nn.Module):
    def __init__(self,
                 diffusion_backbone,
                 forward_SDE=None,
                 training_loss_fn=None,
                 training_time_sampler=None,
                 training_time_uncertainty_sampler=None):
        """
        This is an abstract base class for diffusion models.
        """

        assert isinstance(diffusion_backbone, torch.nn.Module)

        super(DiffusionModel, self).__init__()

        if forward_SDE is None:
            forward_SDE = StandardWienerSDE()

        if training_loss_fn is None:
            training_loss_fn = torch.nn.MSELoss()

        if training_time_sampler is None:
            training_time_sampler = UniformRandomVariable(0.0, 1.0)

        if training_time_uncertainty_sampler is None:
            class IdentitySampler(Sampler):
                def sample(self, t):
                    return t
            training_time_uncertainty_sampler = IdentitySampler()

        assert isinstance(forward_SDE, LinearSDE)
        assert isinstance(diffusion_backbone, torch.nn.Module)
        assert isinstance(training_time_sampler, Sampler)
        assert isinstance(training_loss_fn, torch.nn.Module)

        self.diffusion_backbone = diffusion_backbone
        self.forward_SDE = forward_SDE
        self.training_loss_fn = training_loss_fn
        self.training_time_sampler = training_time_sampler
        self.training_time_uncertainty_sampler = training_time_uncertainty_sampler


    # use union of two data types for typing
    def forward(self, batch_data: torch.Tensor | list):
        """
        This method implements the training loss closure of the diffusion model.
        It computes the loss between the predicted x_0 and the true x_0.
        parameters:
            batch_data: torch.Tensor or list
                The input tensor to the linear operator.
                If it is a tensor, it is assumed to be the true x_0.
                If it is a list, it is assumed to be a tuple of (x_0, y).
        returns:
            loss: torch.Tensor
                The loss between the predicted x_0 and the true x_0.
        """

        if isinstance(batch_data, torch.Tensor):
            x_0 = batch_data
            y=None
        elif isinstance(batch_data, list):
            assert len(batch_data) == 2, "batch_data should a tensor (unconditional) or a tuple/list of two elements (conditional)"
            x_0 = batch_data[0]
            y = batch_data[1]
        else:
            raise ValueError("batch_data should a tensor (unconditional) or a tuple/list of two elements (conditional)")

        batch_size = x_0.shape[0]

        t = self.training_time_sampler.sample(batch_size).to(x_0.device)
        tau = self.training_time_uncertainty_sampler.sample(t).to(x_0.device)
        x_t = self.forward_SDE.sample_x_t_given_x_0(x_0, tau)
        x_0_pred = self.predict_x_0(x_t, t, y)
        loss = self.training_loss_fn(x_0, x_0_pred, t)
        return loss
        
    def sample_reverse_process(self, x_t, timesteps, sampler='euler', return_all=False, y=None, verbose=False):
        """
        This method samples from the reverse SDE.

        parameters:
            x_t: torch.Tensor
                The initial condition.
            timesteps: int
                The number of timesteps to sample.
            sampler: str
                The method used to compute the forward update. Currently, only 'euler' and 'heun' are supported.
            return_all: bool
                If True, the method returns all intermediate samples.
            y: torch.Tensor
                The conditional input to the reverse SDE.
        returns:
            x: torch.Tensor
                The output tensor.
        """
        # we assume the diffusion_backbone estimates the posterior mean of x_0 given x_t, t, and y
        def mean_estimator(x_t, t):
            return self.predict_x_0(x_t, t, y)

        # define the reverse SDE, based on the mean estimator,
        # Tweedies's formula to get the score function, 
        # Anderson's formula to get the reverse SDE
        reverse_SDE = self.forward_SDE.reverse_SDE_given_mean_estimator(mean_estimator)

        return reverse_SDE.sample(x_t, timesteps, sampler, return_all, verbose)
    
    def sample_reverse_process_DPS(self,    
                                x_t, 
                                log_likelihood_fn, 
                                timesteps, 
                                likelihood_weight=1.0,
                                jacobian_method='Backpropagation', # Backpropagation or Identity
                                sampler='euler', 
                                return_all=False, 
                                y=None, 
                                verbose=False):
        """
        Samples from the reverse SDE using diffusion posterior sampling (DPS).

        Parameters:
            x_t: torch.Tensor
                The initial condition.
            log_likelihood_fn: Callable
                Function that returns scalar log-likelihood given x_0.
            timesteps: int
                Number of timesteps to sample.
            likelihood_weight: float
                Weight on the likelihood score.
            jacobian_method: str
                One of ['Backpropagation', 'Identity']
            sampler: str
                Sampler method: 'euler' or 'heun'.
            return_all: bool
                If True, return all intermediate samples.
            y: torch.Tensor
                Optional conditional input.
        Returns:
            x: torch.Tensor
                Final output tensor.
        """

        def prior_score_estimator(x_t, t, x_0):
            Sigma_t = self.forward_SDE.Sigma(t)
            Sigma_t_inv = Sigma_t.inverse_LinearSystem()
            return Sigma_t_inv @ (x_0 - x_t)

        def posterior_score_estimator(x_t, t):
            x_t = x_t.detach().requires_grad_(True)
            x_0_pred = self.predict_x_0(x_t, t, y)
            x_0_pred.retain_grad()

            prior_score = prior_score_estimator(x_t, t, x_0_pred)

            # Zero any old gradients
            if x_t.grad is not None:
                x_t.grad.zero_()
            if x_0_pred.grad is not None:
                x_0_pred.grad.zero_()

            if jacobian_method == 'Backpropagation':
                log_likelihood = log_likelihood_fn(x_0_pred)
                log_likelihood.backward()
                if x_t.grad is None:
                    raise RuntimeError("x_t.grad is None. Did predict_x_0 disconnect x_t from the graph?")
                likelihood_score = x_t.grad.clone()
            elif jacobian_method == 'Identity':
                # Detach x_0_pred so backward does not go through predict_x_0
                x_0_pred_detached = x_0_pred.detach().requires_grad_(True)
                x_0_pred_detached.retain_grad()
                log_likelihood = log_likelihood_fn(x_0_pred_detached)
                log_likelihood.backward()
                if x_0_pred_detached.grad is None:
                    raise RuntimeError("x_0_pred_detached.grad is None. Check that x_0_pred_detached.requires_grad=True.")
                likelihood_score = x_0_pred_detached.grad.clone()
            else:
                raise ValueError(f"Unsupported jacobian_method: {jacobian_method}")

            posterior_score = prior_score + likelihood_weight * likelihood_score
            return posterior_score

        reverse_SDE = self.forward_SDE.reverse_SDE_given_score_estimator(posterior_score_estimator)
        return reverse_SDE.sample(x_t, timesteps, sampler, return_all, verbose)

            
    def predict_x_0(self, x_t: torch.Tensor, t: torch.Tensor, y=None):
        """
        This method predicts x_0 given x_t.

        parameters:
            x_t: torch.Tensor
                The sample at time t.
            t: float
                The time step.
        returns:
            x_0: torch.Tensor
                The predicted initial condition.
        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)

        if y is None:
            x_0_pred =  self.diffusion_backbone(x_t, t)
        else:
            x_0_pred =  self.diffusion_backbone(x_t, t, y)

        return x_0_pred

    def train_diffusion_model(self, 
                             dataset,
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
                             test_plot_vmin=0,
                             test_plot_vmax=1,
                             test_save_plots=True,
                             final_test_iterations='all',
                             # Reverse process sampling parameters
                             reverse_t_start=1.0,
                             reverse_t_end=0.0,
                             reverse_spacing='linear',
                             reverse_sampler='euler',
                             reverse_timesteps=50,
                             **kwargs):
        """
        Train the diffusion model using the provided dataset and training parameters.
        
        Args:
            dataset: Training dataset
            val_data: Validation dataset (optional)
            test_data: Test dataset (optional)
            train_batch_size: Batch size for training
            val_batch_size: Batch size for validation
            test_batch_size: Batch size for testing
            train_num_workers: Number of workers for training dataloader
            val_num_workers: Number of workers for validation dataloader
            test_num_workers: Number of workers for test dataloader
            shuffle_train: Whether to shuffle training data
            shuffle_val: Whether to shuffle validation data
            shuffle_test: Whether to shuffle test data
            num_epochs: Number of training epochs
            num_iterations_train: Number of iterations per training epoch
            num_iterations_val: Number of iterations per validation epoch
            num_iterations_test: Number of iterations per test epoch
            learning_rate: Learning rate for optimizer
            use_ema: Whether to use exponential moving average
            ema_decay: EMA decay rate
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            val_loss_smoothing: Validation loss smoothing factor
            min_delta: Minimum change for early stopping
            verbose: Whether to print verbose output
            very_verbose: Whether to print very verbose output
            wandb_project: WandB project name
            wandb_config: WandB configuration
            save_checkpoints: Whether to save model checkpoints
            experiment_name: Name of the experiment
            output_dir: Output directory for saving results
            epochs_per_evaluation: How often to evaluate
            test_plot_vmin: Minimum value for test plots
            test_plot_vmax: Maximum value for test plots
            test_save_plots: Whether to save test plots
            final_test_iterations: Number of iterations for final test
            reverse_t_start: Starting time for reverse process sampling
            reverse_t_end: Ending time for reverse process sampling
            reverse_spacing: Spacing scheme for timesteps ('linear' or 'logarithmic')
            reverse_sampler: Sampler method for reverse process ('euler' or 'heun')
            reverse_timesteps: Number of timesteps for reverse process
            **kwargs: Additional keyword arguments
            
        Returns:
            Tuple of (train_losses, val_losses, eval_metrics)
        """
        from ..train import train
        import torch.utils.data as data
        from pathlib import Path
        import yaml
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path("gmi_data/outputs")
        else:
            output_dir = Path(output_dir)
        
        if experiment_name is None:
            experiment_name = "unnamed_diffusion_experiment"
        
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'experiment_name': experiment_name,
            'train_dataset': str(type(dataset)),
            'val_dataset': str(type(val_data)) if val_data is not None else None,
            'test_dataset': str(type(test_data)) if test_data is not None else None,
            'diffusion_backbone': str(type(self.diffusion_backbone)),
            'forward_SDE': str(type(self.forward_SDE)),
            'training_loss_fn': str(type(self.training_loss_fn)),
            'training_time_sampler': str(type(self.training_time_sampler)),
            'training_time_uncertainty_sampler': str(type(self.training_time_uncertainty_sampler)),
            'training': {
                'num_epochs': num_epochs,
                'num_iterations_train': num_iterations_train,
                'num_iterations_val': num_iterations_val,
                'num_iterations_test': num_iterations_test,
                'learning_rate': learning_rate,
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'shuffle_train': shuffle_train,
                'shuffle_val': shuffle_val,
                'shuffle_test': shuffle_test,
                'use_ema': use_ema,
                'ema_decay': ema_decay,
                'early_stopping': early_stopping,
                'patience': patience,
                'val_loss_smoothing': val_loss_smoothing,
                'min_delta': min_delta,
                'verbose': verbose,
                'very_verbose': very_verbose,
                'wandb_project': wandb_project,
                'wandb_config': wandb_config,
                'save_checkpoints': save_checkpoints,
                'test_plot_vmin': test_plot_vmin,
                'test_plot_vmax': test_plot_vmax,
                'test_save_plots': test_save_plots,
                'final_test_iterations': final_test_iterations,
                'reverse_t_start': reverse_t_start,
                'reverse_t_end': reverse_t_end,
                'reverse_spacing': reverse_spacing,
                'reverse_sampler': reverse_sampler,
                'reverse_timesteps': reverse_timesteps
            }
        }
        
        config_save_path = experiment_dir / "final_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        print(f"Saved configuration to: {config_save_path}")
        
        # Create test closure for evaluation
        class TestClosure(torch.nn.Module):
            def __init__(self, parent_model, num_iterations_test, test_save_plots, 
                         experiment_dir, reverse_timesteps, reverse_sampler, 
                         test_plot_vmin, test_plot_vmax):
                super().__init__()
                self.parent = parent_model
                self.num_iterations_test = num_iterations_test
                self.test_save_plots = test_save_plots
                self.experiment_dir = experiment_dir
                self.reverse_timesteps = reverse_timesteps
                self.reverse_sampler = reverse_sampler
                self.test_plot_vmin = test_plot_vmin
                self.test_plot_vmax = test_plot_vmax
            
            def forward(self, batch_data, epoch=None, iteration=None):
                """Test closure for diffusion model evaluation."""
                self.eval()
                
                with torch.no_grad():
                    if isinstance(batch_data, (list, tuple)):
                        batch_data = batch_data[0]  # Take only the images for unconditional training
                    
                    batch_data = batch_data.to(next(self.parent.parameters()).device)
                    loss = self.parent(batch_data)
                
                metrics = {
                    'test_loss': loss.item()
                }
                
                # Generate sample images using reverse process
                if self.test_save_plots and epoch is not None and epoch % 10 == 0:  # Save every 10 epochs
                    try:
                        # Sample from the model using reverse process
                        device = next(self.parent.parameters()).device
                        batch_size = 4
                        x_t = torch.randn(batch_size, *batch_data.shape[1:], device=device)
                        
                        # Sample from reverse process with configurable parameters
                        samples = self.parent.sample_reverse_process(
                            x_t, 
                            timesteps=self.reverse_timesteps, 
                            sampler=self.reverse_sampler, 
                            return_all=False,
                            verbose=False
                        )
                        
                        # Save samples
                        import matplotlib.pyplot as plt
                        import numpy as np
                        
                        samples_np = samples.cpu().detach().numpy()
                        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
                        for i in range(4):
                            row, col = i // 2, i % 2
                            # Handle RGB images properly - transpose from (C, H, W) to (H, W, C) for matplotlib
                            if samples_np.shape[1] == 1:  # Grayscale
                                img = samples_np[i, 0]
                                axes[row, col].imshow(img, cmap='gray', vmin=self.test_plot_vmin, vmax=self.test_plot_vmax)
                            else:  # RGB
                                img = samples_np[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                                img = np.clip(img, 0, 1)  # Clip to [0, 1] range for proper display
                                axes[row, col].imshow(img, vmin=self.test_plot_vmin, vmax=self.test_plot_vmax)
                            axes[row, col].axis('off')
                            axes[row, col].set_title(f'Sample {i+1}')
                        
                        plt.tight_layout()
                        plot_path = self.experiment_dir / f"test_samples_epoch_{epoch}.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        # Log to WandB if available
                        try:
                            import wandb
                            # Log the final generated image to WandB
                            wandb.log({
                                f"test_samples_epoch_{epoch}": wandb.Image(plot_path),
                                "epoch": epoch
                            })
                        except ImportError:
                            pass  # WandB not available
                        
                        metrics['test_samples_plot_path'] = str(plot_path)
                        
                    except Exception as e:
                        print(f"Warning: Could not generate test samples: {e}")
                
                return metrics
        
        # Create test closure instance
        test_closure = TestClosure(
            parent_model=self,
            num_iterations_test=num_iterations_test,
            test_save_plots=test_save_plots,
            experiment_dir=experiment_dir,
            reverse_timesteps=reverse_timesteps,
            reverse_sampler=reverse_sampler,
            test_plot_vmin=test_plot_vmin,
            test_plot_vmax=test_plot_vmax
        )
        
        # Setup model saving
        model_to_save = None
        save_best_model_path = None
        if save_checkpoints:
            save_best_model_path = str(experiment_dir / "best_model.pth")
            model_to_save = self.diffusion_backbone
        
        # Train the model using the updated train function
        train_losses, val_losses, eval_metrics = train(
            train_data=dataset,
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
            train_loss_closure=self,
            val_loss_closure=self,
            test_closure=test_closure,
            num_epochs=num_epochs,
            num_iterations=num_iterations_train,
            num_iterations_val=num_iterations_val,
            num_iterations_test=num_iterations_test,
            final_test_iterations=None,  # Disable final test in training function
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
            experiment_name=experiment_name,
            **kwargs
        )
        
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
                    verbose=verbose,
                    reverse_timesteps=reverse_timesteps,
                    reverse_sampler=reverse_sampler,
                    test_plot_vmin=test_plot_vmin,
                    test_plot_vmax=test_plot_vmax
                )
                print("Final evaluation completed successfully!")
            except Exception as e:
                print(f"Warning: Final evaluation failed: {e}")
        
        return train_losses, val_losses, eval_metrics

    def run_final_evaluation(self, test_data, output_dir=None, experiment_name=None, verbose=True,
                           reverse_timesteps=50, reverse_sampler='euler', test_plot_vmin=0, test_plot_vmax=1):
        """
        Run final evaluation on the full test dataset and save per-sample metrics.
        
        Args:
            test_data: Test dataset to evaluate on
            output_dir: Output directory for saving results
            experiment_name: Experiment name for file naming
            verbose: Whether to print progress
            reverse_timesteps: Number of timesteps for reverse process sampling
            reverse_sampler: Sampler method for reverse process
            test_plot_vmin: Minimum value for test plots
            test_plot_vmax: Maximum value for test plots
            
        Returns:
            Dictionary containing summary statistics
        """
        import numpy as np
        import csv
        from pathlib import Path
        from tqdm import tqdm
        import torch.utils.data
        
        # Set device
        device = next(self.parameters()).device
        
        # Create DataLoader if needed
        if isinstance(test_data, torch.utils.data.Dataset):
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)
        elif isinstance(test_data, torch.utils.data.DataLoader):
            test_loader = test_data
        else:
            raise ValueError("test_data must be a PyTorch Dataset or DataLoader")
        
        print(f"\nRunning final evaluation on {len(test_loader)} samples...")
        
        per_sample_metrics = []
        
        # Put model in eval mode
        self.eval()
        
        with torch.no_grad():
            for i, batch_data in enumerate(tqdm(test_loader, desc="Final Evaluation")):
                # Handle different data formats
                if isinstance(batch_data, (tuple, list)):
                    images = batch_data[0].to(device) if isinstance(batch_data[0], torch.Tensor) else batch_data[0]
                elif isinstance(batch_data, torch.Tensor):
                    images = batch_data.to(device)
                else:
                    continue
                
                # Compute loss for each sample in the batch
                for j in range(images.shape[0]):
                    img = images[j:j+1]
                    
                    # Compute loss
                    loss = self(img)
                    
                    per_sample_metrics.append({
                        'index': len(per_sample_metrics),
                        'test_loss': loss.item()
                    })
        
        # Save per-sample metrics to CSV
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            per_sample_csv = output_path / "final_evaluation_metrics_per_sample.csv"
            with open(per_sample_csv, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['index', 'test_loss'])
                writer.writeheader()
                for row in per_sample_metrics:
                    writer.writerow(row)
            
            if verbose:
                print(f"Per-sample metrics saved to: {per_sample_csv}")
        
        # Compute summary statistics
        summary_stats = {}
        if per_sample_metrics:
            values = np.array([row['test_loss'] for row in per_sample_metrics])
            summary_stats['test_loss_mean'] = float(np.mean(values))
            summary_stats['test_loss_std'] = float(np.std(values))
            summary_stats['test_loss_min'] = float(np.min(values))
            summary_stats['test_loss_max'] = float(np.max(values))
        
        # Print summary if verbose
        if verbose and per_sample_metrics:
            print(f"\nFinal evaluation summary:")
            values = [row['test_loss'] for row in per_sample_metrics]
            print(f"  test_loss: mean={np.mean(values):.4f}, std={np.std(values):.4f}, min={np.min(values):.4f}, max={np.max(values):.4f}")
        
        return summary_stats

    def train_diffusion_model_from_config(self, config_dict, device=None, experiment_name=None, output_dir=None):
        """
        Train the diffusion model using a configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary containing training parameters
            device: Device to use (default: auto-detect)
            experiment_name: Override experiment name from config
            output_dir: Override output directory from config
            
        Returns:
            Tuple of (train_losses, val_losses, eval_metrics)
        """
        from ..config import load_object_from_dict
        
        # Extract components from config
        train_dataset = config_dict.get('train_dataset')
        validation_dataset = config_dict.get('validation_dataset')
        test_dataset = config_dict.get('test_dataset')
        diffusion_backbone = config_dict.get('diffusion_backbone')
        
        # Extract diffusion-specific components
        forward_SDE = config_dict.get('forward_SDE')
        training_loss_fn = config_dict.get('training_loss_fn')
        training_time_sampler = config_dict.get('training_time_sampler')
        training_time_uncertainty_sampler = config_dict.get('training_time_uncertainty_sampler')
        
        # Load components if they are config dictionaries
        if isinstance(train_dataset, dict):
            train_dataset = load_object_from_dict(train_dataset)
        if isinstance(validation_dataset, dict):
            validation_dataset = load_object_from_dict(validation_dataset)
        if isinstance(test_dataset, dict):
            test_dataset = load_object_from_dict(test_dataset)
        if isinstance(diffusion_backbone, dict):
            diffusion_backbone = load_object_from_dict(diffusion_backbone)
        
        # Load diffusion-specific components if they are config dictionaries
        if isinstance(forward_SDE, dict):
            forward_SDE = load_object_from_dict(forward_SDE)
        if isinstance(training_loss_fn, dict):
            training_loss_fn = load_object_from_dict(training_loss_fn)
        if isinstance(training_time_sampler, dict):
            training_time_sampler = load_object_from_dict(training_time_sampler)
        if isinstance(training_time_uncertainty_sampler, dict):
            training_time_uncertainty_sampler = load_object_from_dict(training_time_uncertainty_sampler)
        
        # Use train_dataset as the main dataset
        dataset = train_dataset
        
        if not all([dataset, diffusion_backbone]):
            raise ValueError("Configuration must contain 'train_dataset' and 'diffusion_backbone'")
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Using device: {device}")
        if device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Move diffusion backbone to device if it's not already
        if hasattr(diffusion_backbone, 'to'):
            diffusion_backbone = diffusion_backbone.to(device)
        
        # Get experiment name
        if experiment_name is None:
            experiment_name = config_dict.get('experiment_name', 'unnamed_diffusion_experiment')
        
        # Determine output directory
        if output_dir is None:
            output_dir = config_dict.get('output_dir', f"gmi_data/outputs/{experiment_name}")
        
        print(f"Starting training for experiment: {experiment_name}")
        
        # Create diffusion model with optional components
        diffusion_model = DiffusionModel(
            diffusion_backbone=diffusion_backbone,
            forward_SDE=forward_SDE,
            training_loss_fn=training_loss_fn,
            training_time_sampler=training_time_sampler,
            training_time_uncertainty_sampler=training_time_uncertainty_sampler
        )
        
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
        num_iterations_val = to_int(training_config.get('num_iterations_val', 10), 10)
        num_iterations_test = to_int(training_config.get('num_iterations_test', 1), 1)
        learning_rate = to_float(training_config.get('learning_rate', 0.001), 0.001)
        batch_size = to_int(training_config.get('batch_size', 4), 4)
        num_workers = to_int(training_config.get('num_workers', 4), 4)
        patience = to_int(training_config.get('patience', 10), 10)
        val_loss_smoothing = to_float(training_config.get('val_loss_smoothing', 0.9), 0.9)
        min_delta = to_float(training_config.get('min_delta', 1e-6), 1e-6)
        ema_decay = to_float(training_config.get('ema_decay', 0.999), 0.999)
        test_plot_vmin = to_float(training_config.get('test_plot_vmin', 0), 0)
        test_plot_vmax = to_float(training_config.get('test_plot_vmax', 1), 1)
        reverse_t_start = to_float(training_config.get('reverse_t_start', 1.0), 1.0)
        reverse_t_end = to_float(training_config.get('reverse_t_end', 0.0), 0.0)
        reverse_timesteps = to_int(training_config.get('reverse_timesteps', 50), 50)
        
        # Train the model with training config
        train_losses, val_losses, eval_metrics = diffusion_model.train_diffusion_model(
            dataset=dataset,
            val_data=validation_dataset,
            test_data=test_dataset,
            experiment_name=experiment_name,
            output_dir=output_dir,
            num_epochs=num_epochs,
            num_iterations_train=num_iterations_train,
            num_iterations_val=num_iterations_val,
            num_iterations_test=num_iterations_test,
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
            verbose=training_config.get('verbose', True),
            very_verbose=training_config.get('very_verbose', False),
            wandb_project=training_config.get('wandb_project', None),
            wandb_config=training_config.get('wandb_config', None),
            save_checkpoints=training_config.get('save_checkpoints', True),
            test_plot_vmin=test_plot_vmin,
            test_plot_vmax=test_plot_vmax,
            test_save_plots=training_config.get('test_save_plots', True),
            final_test_iterations=training_config.get('final_test_iterations', 'all'),
            reverse_t_start=reverse_t_start,
            reverse_t_end=reverse_t_end,
            reverse_spacing=training_config.get('reverse_spacing', 'linear'),
            reverse_sampler=training_config.get('reverse_sampler', 'euler'),
            reverse_timesteps=reverse_timesteps
        )
        
        print(f"Training completed!")
        print(f"Final train loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"Final validation loss: {val_losses[-1]:.4f}")
        
        return train_losses, val_losses, eval_metrics

    @classmethod
    def train_from_config_file(cls, config_path, device=None, experiment_name=None, output_dir=None):
        """
        Train a diffusion model from a YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            device: Device to use (default: auto-detect)
            experiment_name: Override experiment name from config
            output_dir: Override output directory from config
            
        Returns:
            Tuple of (train_losses, val_losses, eval_metrics)
        """
        import yaml
        from pathlib import Path
        from ..config import load_object_from_dict
        
        # Load configuration file
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract diffusion backbone from config
        diffusion_backbone = config_dict.get('diffusion_backbone')
        if diffusion_backbone is None:
            raise ValueError("Configuration must contain 'diffusion_backbone'")
        
        # Load diffusion backbone if it's a config dictionary
        if isinstance(diffusion_backbone, dict):
            diffusion_backbone = load_object_from_dict(diffusion_backbone)
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Move diffusion backbone to device
        if hasattr(diffusion_backbone, 'to'):
            diffusion_backbone = diffusion_backbone.to(device)
        
        # Create diffusion model
        diffusion_model = cls(diffusion_backbone=diffusion_backbone)
        
        # Train using the config
        return diffusion_model.train_diffusion_model_from_config(
            config_dict, device, experiment_name, output_dir
        )

class DiffusionBackbone(torch.nn.Module):
    def __init__(self,
                 x_t_encoder,
                 t_encoder,
                 x_0_predictor,
                 y_encoder=None,
                 pass_t_to_x_0_predictor=False,
                 c_skip=None,
                 c_out=None,
                 c_in=None,
                 c_noise=None):
        
        """
        
        This is designed to implement a diffusion backbone. It predicts x_0 given x_t, t, and y embeddings.

        x_t is a sample from the forward or reverse diffusion process at time t, it is a tensor of shape [batch_size, *x_t.shape]
        t is the time step. We assume it is a tensor of shape [batch_size, 1]
        y is an optional conditional input to the diffusion backbone, it is a tensor of shape [batch_size, *y.shape]


        parameters:
            x_t_encoder: torch.nn.Module
                The neural network that encodes information from x_t.
            t_encoder: torch.nn.Module
                The neural network that encodes information from t.
            x_0_predictor: torch.nn.Module
                The neural network that predicts x_0 given x_t, t, and y embeddings.
            y_encoder: torch.nn.Module
                The optional neural network that encodes information from y.
        """
        
        assert isinstance(x_t_encoder, torch.nn.Module)
        assert isinstance(t_encoder, torch.nn.Module)
        assert isinstance(x_0_predictor, torch.nn.Module)

        if y_encoder is not None:
            assert isinstance(y_encoder, torch.nn.Module)

        super(DiffusionBackbone, self).__init__()

        self.x_t_encoder = x_t_encoder
        self.t_encoder = t_encoder
        self.x_0_predictor = x_0_predictor
        self.y_encoder = y_encoder

        self.pass_t_to_x_0_predictor = pass_t_to_x_0_predictor

        def expand_t_to_x_shape(t, x_shape):
            assert isinstance(t, torch.Tensor)
            assert t.shape[0] == x_shape[0], f"t.shape[0] = {t.shape[0]}, x_shape[0] = {x_shape[0]}"
            t_shape = [t.shape[0]] + [1]*len(x_shape[1:])
            t = t.reshape(t_shape)
            return t

        if c_skip is None:
            c_skip = lambda t,x_shape: 0.0
        
        if c_out is None:
            c_out = lambda t,x_shape: 1.0

        if c_in is None:
            c_in = lambda t,x_shape: 1.0
            
        if c_noise is None:
            c_noise = lambda t,x_shape: 1.0

        self.c_skip = c_skip
        self.c_out = c_out
        self.c_in = c_in
        self.c_noise = c_noise

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y=None):

        """
        This method implements the forward pass of the diffusion backbone.

        """

        assert isinstance(x_t, torch.Tensor)
        assert isinstance(t, torch.Tensor)
        if y is not None:
            assert isinstance(y, torch.Tensor)
        
        x_t_embedding = self.x_t_encoder(x_t)
        t_embedding = self.t_encoder(t)

        if y is not None and self.y_encoder is not None:
                y_embedding = self.y_encoder(y)

        if self.pass_t_to_x_0_predictor:
            if y is not None and self.y_encoder is not None:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, y_embedding, (self.c_noise(t,t.shape)*t).squeeze())
            else:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, (self.c_noise(t,t.shape)*t).squeeze())
        else:
            if y is not None and self.y_encoder is not None:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding, y_embedding)
            else:
                x_out = self.x_0_predictor(self.c_in(t,x_t_embedding.shape)*x_t_embedding, t_embedding)


        x_0_pred = self.c_skip(t,x_t.shape)*x_t + self.c_out(t,x_out.shape)*x_out

        return x_0_pred



