"""
Newton's Method optimizer for CT inverse problems.

This module provides a flexible Newton's method optimizer for CT applications
including beam hardening correction, thickness recovery, and spectral decomposition.
"""

import torch
import numpy as np
from typing import Callable, Optional, Dict, Any, List


class NewtonOptimizer:
    """
    Newton's Method optimizer for Maximum Likelihood Beam Hardening Correction.
    
    This optimizer uses second-order Newton's method to minimize the negative
    log-likelihood (NLL) for various CT inverse problems. It supports:
    - Single parameter optimization (e.g., thickness estimation)
    - Batched optimization (e.g., bias analysis across multiple models/thicknesses)
    - Multi-dimensional optimization (e.g., PE/CS decomposition)
    
    The forward model should provide (mean, variance) predictions given parameters.
    
    Example:
        >>> def forward_fn(x):
        >>>     # x: [batch_size, n_params] parameters to optimize
        >>>     # Returns: (mu, var) each [batch_size, n_channels]
        >>>     mu, var = xray_system.forward_stats(x)
        >>>     return mu, var
        >>> 
        >>> optimizer = NewtonMLBHC(
        >>>     forward_fn=forward_fn,
        >>>     y_obs=measurements,
        >>>     x_init=initial_guess,
        >>>     learning_rate=0.9,
        >>>     max_iters=50
        >>> )
        >>> 
        >>> # Run with callbacks for animation
        >>> def callback(iteration, x_current, nll_current):
        >>>     # Update visualization
        >>>     pass
        >>> 
        >>> x_final = optimizer.optimize(callback=callback)
    """
    
    def __init__(
        self,
        forward_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        y_obs: torch.Tensor,
        x_init: torch.Tensor,
        learning_rate: float = 0.9,
        hessian_reg_factor: float = 0.001,
        hessian_epsilon: float = 1e-9,
        max_iters: int = 50,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        device: Optional[torch.device] = None,
        use_log_prob: bool = False,
        log_prob_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Initialize Newton's method optimizer.
        
        Args:
            forward_fn: Function that takes parameters x and returns (mu, var).
                       x shape: [batch_size, n_params]
                       mu, var shape: [batch_size, n_channels]
            y_obs: Observed measurements [batch_size, n_channels]
            x_init: Initial parameter guess [batch_size, n_params]
            learning_rate: Step size multiplier (default: 0.9)
            hessian_reg_factor: Hessian regularization: h_reg = h + factor * |h| (default: 0.001)
            hessian_epsilon: Small constant for numerical stability (default: 1e-9)
            max_iters: Maximum number of Newton iterations (default: 50)
            x_min: Optional lower bound for parameters
            x_max: Optional upper bound for parameters
            device: Torch device (default: inferred from x_init)
            use_log_prob: If True, use log_prob_fn instead of manual NLL computation (default: False)
            log_prob_fn: Optional log probability function log_prob_fn(x, y_obs) -> log_prob
                        Used when use_log_prob=True
        """
        self.forward_fn = forward_fn
        self.y_obs = y_obs
        self.learning_rate = learning_rate
        self.hessian_reg_factor = hessian_reg_factor
        self.hessian_epsilon = hessian_epsilon
        self.max_iters = max_iters
        self.x_min = x_min
        self.x_max = x_max
        self.device = device if device is not None else x_init.device
        self.use_log_prob = use_log_prob
        self.log_prob_fn = log_prob_fn
        
        # Initialize parameters
        self.x = x_init.clone().to(self.device).requires_grad_(True)
        
        # History tracking
        self.history: List[Dict[str, Any]] = []
        
    def compute_nll(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood for current parameters.
        
        If use_log_prob=True and log_prob_fn is provided, uses that.
        Otherwise: NLL = 0.5 * sum((y_obs - mu)^2 / var + log(var))
        
        Args:
            x: Parameters [batch_size, n_params]
            
        Returns:
            nll: Scalar negative log-likelihood (or -log_prob)
        """
        if self.use_log_prob and self.log_prob_fn is not None:
            # Use provided log_prob function (returns log probability)
            log_prob = self.log_prob_fn(x, self.y_obs)
            # If any point is NaN, it affects the total sum. 
            # NewtonOptimizer's NLL is usually a sum across the batch.
            return -log_prob.sum()
        else:
            # Manual NLL computation from forward_stats
            mu, var = self.forward_fn(x)
            nll = 0.5 * (((self.y_obs - mu)**2 / var) + torch.log(var))
            return nll.sum()
    
    def compute_hessian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian (second derivative) of NLL with respect to parameters.
        
        Args:
            x: Parameters [batch_size, n_params]
            
        Returns:
            hess: Hessian [batch_size, n_params]
        """
        # Compute NLL
        nll = self.compute_nll(x)
        
        # Compute gradient
        grad = torch.autograd.grad(nll, x, create_graph=True)[0]
        
        # Compute Hessian (second derivative)
        hess = torch.autograd.grad(grad.sum(), x)[0]
        
        return hess
    
    def auto_newton_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard Newton step using the diagonal of the Hessian (current implementation).
        """
        # Compute NLL
        nll = self.compute_nll(self.x)
        
        # Compute gradient
        grad = torch.autograd.grad(nll, self.x, create_graph=True)[0]
        
        # Compute Hessian (diagonal elements)
        hess = torch.autograd.grad(grad.sum(), self.x)[0]
        
        # Apply Newton update with regularization
        with torch.no_grad():
            # Regularize Hessian to ensure stability
            hess_reg = hess + self.hessian_reg_factor * torch.abs(hess)
            
            # Newton step (diagonal only)
            self.x -= self.learning_rate * grad / (hess_reg + self.hessian_epsilon)
            
            # Apply bounds if specified
            if self.x_min is not None or self.x_max is not None:
                self.x.clamp_(min=self.x_min, max=self.x_max)
        
        return grad, hess

    def manual_2x2_newton_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full 2x2 Newton step for dual-energy problems.
        Correctly handles off-diagonal Hessian terms.
        """
        # Compute NLL
        nll = self.compute_nll(self.x)
        
        # Gradient
        grads = torch.autograd.grad(nll, self.x, create_graph=True)[0]
        # x shape: [batch_size, 2]
        # grads shape: [batch_size, 2]
        
        # Get individual gradient components
        g1 = grads[:, 0]
        g2 = grads[:, 1]
        
        # Second derivatives (Hessian) for each component
        # We need h11, h12, h21, h22 where hij = d^2(nll)/dxi dxj
        # d/dx1 (g1.sum()) -> [h11, h12]
        h1 = torch.autograd.grad(g1.sum(), self.x, retain_graph=True)[0]
        h11 = h1[:, 0]
        h12 = h1[:, 1]
        
        # d/dx2 (g2.sum()) -> [h21, h22]
        h2 = torch.autograd.grad(g2.sum(), self.x)[0]
        h21 = h2[:, 0]
        h22 = h2[:, 1]

        # Force symmetry
        h12_avg = (h12 + h21) / 2.0
        
        # Newton update with regularization
        with torch.no_grad():
            # Apply regularization:
            h11_reg = h11 + self.hessian_reg_factor * torch.abs(h11) + self.hessian_epsilon
            h22_reg = h22 + self.hessian_reg_factor * torch.abs(h22) + self.hessian_epsilon
            h12_reg = h12_avg + self.hessian_reg_factor * torch.abs(h12_avg)
            
            # Determinant of the regularized Hessian
            det = h11_reg * h22_reg - h12_reg**2
            
            # Matrix inversion for 2x2
            inv_h11 = h22_reg / det
            inv_h12 = -h12_reg / det
            inv_h22 = h11_reg / det
            
            # Newton direction: dx = H^-1 * g
            dx1 = inv_h11 * g1 + inv_h12 * g2
            dx2 = inv_h12 * g1 + inv_h22 * g2
            
            # Update parameters
            self.x.data[:, 0] -= self.learning_rate * dx1
            self.x.data[:, 1] -= self.learning_rate * dx2
            
            # Apply bounds
            if self.x_min is not None or self.x_max is not None:
                self.x.data.clamp_(min=self.x_min, max=self.x_max)
                
        # Return gradient and a pseudo-hessian (diagonal only for consistent API)
        return grads, torch.stack([h11, h22], dim=-1)


    def newton_step(self) -> tuple[torch.Tensor, torch.Tensor]:
        # If dual-energy (n_materials=2), use full 2x2. Otherwise diagonal.
        if self.x.shape[1] == 2:
            return self.manual_2x2_newton_step()
        else:
            return self.auto_newton_step()
    
    def optimize(
        self,
        callback: Optional[Callable[[int, torch.Tensor, float], None]] = None,
        store_history: bool = True,
    ) -> torch.Tensor:
        """
        Run Newton's method optimization.
        
        Args:
            callback: Optional callback function called after each iteration.
                     Signature: callback(iteration: int, x_current: Tensor, nll_current: float)
                     Useful for animations and progress tracking.
            store_history: Whether to store optimization history (default: True)
            
        Returns:
            x_final: Optimized parameters [batch_size, n_params]
        """
        for iteration in range(self.max_iters):
            # Compute current NLL for callback/history
            with torch.no_grad():
                nll_current = self.compute_nll(self.x).item()
            
            # Store history
            if store_history:
                self.history.append({
                    'iteration': iteration,
                    'x': self.x.detach().clone(),
                    'nll': nll_current,
                })
            
            # Call user callback
            if callback is not None:
                callback(iteration, self.x.detach().clone(), nll_current)
            
            # Perform Newton step
            grad, hess = self.newton_step()
        
        # Final state
        with torch.no_grad():
            nll_final = self.compute_nll(self.x).item()
        
        if store_history:
            self.history.append({
                'iteration': self.max_iters,
                'x': self.x.detach().clone(),
                'nll': nll_final,
            })
        
        return self.x.detach()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get optimization history.
        
        Returns:
            history: List of dicts with keys ['iteration', 'x', 'nll']
        """
        return self.history
    
    def precompute_nll_surface(
        self,
        param_ranges: List[np.ndarray],
        resolution: int = 500,
    ) -> np.ndarray:
        """
        Precompute NLL surface for visualization (2D case only).
        
        Args:
            param_ranges: List of [min, max] ranges for each parameter dimension
            resolution: Grid resolution per dimension (default: 500)
            
        Returns:
            nll_surface: NLL values on grid [resolution, resolution] for 2D,
                        [resolution,] for 1D
        """
        n_dims = len(param_ranges)
        
        if n_dims == 1:
            # 1D case
            grid = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
            x_batch = torch.from_numpy(grid).float().to(self.device).view(-1, 1)
            
            with torch.no_grad():
                nll_vals = []
                batch_size = 10000
                for i in range(0, x_batch.shape[0], batch_size):
                    x_chunk = x_batch[i:i+batch_size]
                    # Replicate y_obs for batch
                    y_obs_batch = self.y_obs.expand(x_chunk.shape[0], -1)
                    mu, var = self.forward_fn(x_chunk)
                    nll = 0.5 * (((y_obs_batch - mu)**2 / var) + torch.log(var))
                    nll_vals.append(nll.sum(dim=1).cpu().numpy())
                
                nll_surface = np.concatenate(nll_vals)
            
            return nll_surface
            
        elif n_dims == 2:
            # 2D case
            grid_0 = np.linspace(param_ranges[0][0], param_ranges[0][1], resolution)
            grid_1 = np.linspace(param_ranges[1][0], param_ranges[1][1], resolution)
            G0, G1 = np.meshgrid(grid_0, grid_1)
            
            x_batch = torch.stack([
                torch.from_numpy(G0.flatten()).float(),
                torch.from_numpy(G1.flatten()).float()
            ], dim=1).to(self.device)
            
            with torch.no_grad():
                nll_vals = []
                batch_size = 50000
                for i in range(0, x_batch.shape[0], batch_size):
                    x_chunk = x_batch[i:i+batch_size]
                    # Replicate y_obs for batch
                    y_obs_batch = self.y_obs.expand(x_chunk.shape[0], -1)
                    mu, var = self.forward_fn(x_chunk)
                    nll = 0.5 * (((y_obs_batch - mu)**2 / var) + torch.log(var))
                    nll_vals.append(nll.sum(dim=1).cpu().numpy())
                
                nll_surface = np.concatenate(nll_vals).reshape(resolution, resolution)
            
            return nll_surface
        
        else:
            raise ValueError(f"Only 1D and 2D parameter spaces supported for surface computation, got {n_dims}D")
