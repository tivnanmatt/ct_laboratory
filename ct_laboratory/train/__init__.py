import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from typing import Optional
import time



def train(
    train_data,
    train_loss_closure,
    num_epochs=10,
    num_iterations=100,
    optimizer=None,
    lr=1e-3,
    lr_scheduler=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    fabric=None,
    val_data=None,
    val_loss_closure=None,
    num_iterations_val=None,
    verbose=True,
    very_verbose=False,
    early_stopping=False,
    patience=10,
    val_loss_smoothing=0.9,
    min_delta=1e-6,
    use_ema=False,
    ema_decay=0.999
):
    """
    Simplified training function for any model with an Adam optimizer using a provided loss closure.

    Args:
        train_data (Dataset or DataLoader): Training data. If Dataset, will use batch_size=1, num_workers=1, shuffle=False.
        train_loss_closure (nn.Module): Module for training loss calculation.
        val_data (Dataset or DataLoader, optional): Validation data. If Dataset, will use batch_size=1, num_workers=1, shuffle=False.
        val_loss_closure (nn.Module, optional): Module for validation loss calculation. If None, uses train_loss_closure.
        num_epochs (int): The number of epochs to train for.
        num_iterations (int): The number of iterations per epoch.
        num_iterations_val (int, optional): The number of validation iterations per epoch. If None, uses num_iterations.
        optimizer (torch.optim.Optimizer, optional): Optimizer to use for training. If None, uses Adam.
        lr (float): The learning rate for the Adam optimizer.
        device (str): The device to train on, 'cuda' or 'cpu'.
        fabric (optional): Fabric instance for distributed training.
        early_stopping (bool): Whether to use early stopping based on validation loss.
        patience (int): Number of epochs to wait for improvement before early stopping.
        val_loss_smoothing (float): Smoothing factor for exponential running average of validation loss.
        min_delta (float): Minimum change to qualify as an improvement for early stopping.
        use_ema (bool): If True, applies Exponential Moving Average to model weights.
        ema_decay (float): Decay factor for EMA.
        verbose (bool): Whether to print training progress.
        very_verbose (bool): Whether to print loss for each batch.
    Returns:
        tuple: Lists of average losses recorded at each epoch (train_losses, val_losses).
    """
    import torch.utils.data
    
    # Set default closures
    if val_loss_closure is None:
        val_loss_closure = train_loss_closure
    
    # Helper to wrap dataset in DataLoader if needed
    def make_loader(data):
        if data is None:
            return None
        if isinstance(data, torch.utils.data.DataLoader):
            return data
        # Default values for Dataset
        return torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    train_loader = make_loader(train_data)
    validation_loader = make_loader(val_data)

    if optimizer is None:
        optimizer = torch.optim.Adam(train_loss_closure.parameters(), lr=lr)
    if num_iterations_val is None:
        num_iterations_val = num_iterations
    
    # Initialize EMA if enabled
    ema = ExponentialMovingAverage(train_loss_closure.parameters(), decay=ema_decay) if use_ema else None
    
    # Early stopping variables
    smoothed_val_loss = None
    patience_counter = 0
    best_val_loss = float('inf')

    # Trackers for loss history
    train_losses, val_losses = [], []
    train_loader_iter = iter(train_loader) if train_loader else None
    val_loader_iter = iter(validation_loader) if validation_loader else None

    for epoch in range(num_epochs):
        train_batch_losses = []

        epoch_start_time = time.time()

        for _ in tqdm(range(num_iterations), desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            try:
                if train_loader_iter is None and train_loader is not None:
                    train_loader_iter = iter(train_loader)
                batch_data = next(train_loader_iter) if train_loader_iter is not None else None
            except StopIteration:
                if train_loader is not None:
                    train_loader_iter = iter(train_loader)
                    batch_data = next(train_loader_iter)
                else:
                    batch_data = None

            # Move batch data to device and handle both tensors and tuples
            if isinstance(batch_data, (tuple, list)):
                batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data)
            elif isinstance(batch_data, torch.Tensor):
                batch_data = batch_data.to(device)

            optimizer.zero_grad()
            # Handle both single tensors and tuples by unpacking tuples
            if isinstance(batch_data, (tuple, list)):
                loss = train_loss_closure(*batch_data)
            else:
                loss = train_loss_closure(batch_data)
            
            if fabric is None:
                loss.backward()
            else:
                fabric.backward(loss)
            
            optimizer.step()

            # Apply EMA to weights
            if ema:
                ema.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            train_batch_losses.append(loss.item())
            if very_verbose:
                print(f"Training Batch Loss: {loss.item():.4f}")
        
        epoch_train_time = time.time() - epoch_start_time
        
        # Record average train loss for the epoch
        train_epoch_loss = sum(train_batch_losses) / len(train_batch_losses)
        train_losses.append(train_epoch_loss)

        # Validation phase
        if validation_loader:
            val_batch_losses = []
            with torch.no_grad():
                for _ in tqdm(range(num_iterations_val), desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                    try:
                        if val_loader_iter is None and validation_loader is not None:
                            val_loader_iter = iter(validation_loader)
                        batch_data = next(val_loader_iter) if val_loader_iter is not None else None
                    except StopIteration:
                        if validation_loader is not None:
                            val_loader_iter = iter(validation_loader)
                            batch_data = next(val_loader_iter)
                        else:
                            batch_data = None

                    # Move batch data to device and handle both tensors and tuples (same as training)
                    if isinstance(batch_data, (tuple, list)):
                        batch_data = tuple(d.to(device) if isinstance(d, torch.Tensor) else d for d in batch_data)
                    elif isinstance(batch_data, torch.Tensor):
                        batch_data = batch_data.to(device)
                    
                    # Handle both single tensors and tuples by unpacking tuples
                    if isinstance(batch_data, (tuple, list)):
                        val_loss = val_loss_closure(*batch_data)
                    else:
                        val_loss = val_loss_closure(batch_data)
                    val_batch_losses.append(val_loss.item())

            # Record average validation loss
            val_epoch_loss = sum(val_batch_losses) / len(val_batch_losses)
            val_losses.append(val_epoch_loss)

            # Smooth the validation loss
            if smoothed_val_loss is None:
                smoothed_val_loss = val_epoch_loss
            else:
                smoothed_val_loss = val_loss_smoothing * smoothed_val_loss + (1 - val_loss_smoothing) * val_epoch_loss
            
            # Determine which validation loss to use for early stopping and patience
            # If val_loss_smoothing is 0, use raw validation loss; otherwise use smoothed
            val_loss_for_tracking = val_epoch_loss if val_loss_smoothing == 0 else smoothed_val_loss
            
            # Update best validation loss and patience counter
            if val_loss_for_tracking < best_val_loss - min_delta:
                best_val_loss = val_loss_for_tracking
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping check
            if early_stopping and patience_counter >= patience:
                loss_type = "raw validation loss" if val_loss_smoothing == 0 else "smoothed validation loss"
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in {loss_type}. The {loss_type} is {val_loss_for_tracking:.4f} and the best validation loss is {best_val_loss:.4f}.")
                break

            # Print losses if verbose
            if verbose:
                loss_type = "Raw Val Loss" if val_loss_smoothing == 0 else "Smoothed Val Loss"
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | {loss_type}: {val_loss_for_tracking:.4f} | Best {loss_type}: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_epoch_loss:.4f}")
    
    # Apply EMA weights before returning, if using EMA
    if ema:
        ema.store()
        ema.copy_to()
    
    return train_losses, val_losses
