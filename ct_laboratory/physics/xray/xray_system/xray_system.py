import torch
import numpy as np
from typing import Optional, Tuple, List
from tqdm import tqdm
from ct_laboratory.physics.xray.scatter.scatter_models import ZeroScatterModel
from ct_laboratory.physics.xray.attenuation.operators import ObjectAttenuator
from ct_laboratory.physics.xray.optimization import NewtonOptimizer


class CoordinateTransformer(torch.nn.Module):
    """
    Module for preprocessing log-measurements before lookup.
    Handles applying a linear transformation (e.g., PCA) to rotate
    the coordinate system for the lookup table.
    """
    def __init__(self, rotation_matrix: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor):
        """
        Args:
            rotation_matrix: [n_channels, n_channels] rotation/whitening matrix (e.g. from PCA)
            offset: [1, n_channels] offset to center the data
            scale: [1, n_channels] scaling factor for z-score normalization
        """
        super().__init__()
        self.register_buffer("rotation_matrix", rotation_matrix)
        self.register_buffer("offset", offset)
        self.register_buffer("scale", scale)
        
    def forward(self, y_log: torch.Tensor) -> torch.Tensor:
        """
        Apply coordinate transformation to log-measurements.
        
        y_trans = (y_log - offset) @ rotation_matrix / scale
        """
        y_centered = y_log - self.offset
        y_rotated = y_centered @ self.rotation_matrix
        y_norm = y_rotated / (self.scale + 1e-12)
            
        return y_norm


class DifferentiableLookupTable(torch.nn.Module):
    """
    Differentiable bicubic lookup table for fast material decomposition.
    
    This module implements a differentiable bicubic interpolation lookup table
    that maps dual-energy measurements (y_obs) to material basis coefficients
    using a CoordinateTransformer for preprocessing.
    
    Shape conventions:
        table: [n_ch1, n_ch2, n_basis] - lookup table
        y_obs: [*, n_channels] - measurements
        output: [*, n_basis] - material basis coefficients
    """
    
    def __init__(
        self, 
        table: torch.Tensor, 
        transformer: CoordinateTransformer,
        lut_z_min: torch.Tensor,
        lut_z_max: torch.Tensor,
    ):
        """
        Initialize the differentiable lookup table.
        
        Args:
            table: Pre-computed lookup table [n_ch1, n_ch2, n_basis]
            transformer: Preprocessing module (Log + PCA + Scaling)
            lut_z_min: [1, n_channels] MIN values in transformer space defining the LUT extent
            lut_z_max: [1, n_channels] MAX values in transformer space defining the LUT extent
        """
        super().__init__()
        self.register_buffer("table", table)
        self.transformer = transformer
        self.register_buffer("lut_z_min", lut_z_min)
        self.register_buffer("lut_z_max", lut_z_max)
    
    def forward(self, y_obs: torch.Tensor) -> torch.Tensor:
        """
        Perform differentiable bicubic interpolation lookup.
        
        Args:
            y_obs: Measurements [*, n_channels]
            
        Returns:
            basis_coeffs: Material basis coefficients [*, n_basis]
        """
        orig_shape = y_obs.shape
        y_flat = y_obs.reshape(-1, orig_shape[-1])
        
        # Take log of measurements
        eps = 1e-12
        y_log = -torch.log(torch.clamp(y_flat, min=eps))
        
        # Apply coordinate transformation: y_z = (y_log - offset) @ rotation
        # Note: In PCA mode this is the rotated coordinate. In axis-aligned it is y_log.
        y_centered = y_log - self.transformer.offset
        y_z = y_centered @ self.transformer.rotation_matrix
        
        # Map y_z space to grid_sample range [-1, 1] based on stored lut_z bounds
        # y_norm = 2.0 * (y_z - min) / (max - min) - 1.0
        y_norm = 2.0 * (y_z - self.lut_z_min) / (self.lut_z_max - self.lut_z_min + 1e-12) - 1.0
        
        n_ch1, n_ch2, n_basis = self.table.shape
        
        # grid_sample expects input: [N, C, H, W] and grid: [N, H_out, W_out, 2]
        # Our table is [n_ch1, n_ch2, n_basis] -> [row, col, chan]
        # Permute to [1, n_basis, n_ch1, n_ch2] -> [1, C, H, W]
        lut_tensor = self.table.permute(2, 0, 1).unsqueeze(0)  
        
        # grid_sample coordinates are (x, y) where x is the horizontal (last) dimension
        # In our case, dim 1 is horizontal (n_ch2), dim 0 is vertical (n_ch1)
        # y_norm columns [0, 1] corresponds to vertical [0] and horizontal [1]
        # So we flip for grid_sample: [y_norm[:, 1], y_norm[:, 0]]
        grid = y_norm[:, [1, 0]].view(1, -1, 1, 2)
        
        # Perform bicubic interpolation
        result = torch.nn.functional.grid_sample(
            lut_tensor, grid, 
            mode='bicubic', 
            padding_mode='border',
            align_corners=True
        ).view(n_basis, -1).permute(1, 0)
        
        # Reshape back to original shape plus basis dimension
        return result.view(*orig_shape[:-1], n_basis)


class XraySystem(torch.nn.Module):
    """
    Full spectral CT forward model with Poisson statistics and system blur.
    
    Shape conventions (CONSISTENT TENSOR-ORDERING):
        energies_keV: [n_energies] - energy bins in keV
        basis_materials.Q: [n_energies, n_materials] - basis attenuation spectra (n_materials=2 for PE/CS)
        x_basis:      [n_rays, n_materials] - basis line integrals (ALWAYS [n_rays, n_materials])
        q:            [n_rays, n_exposures, n_energies] - spectral photon distribution at detector
        y:            [n_rays, n_channels] - integrated detector signal (e.g. 2 channels for dual-exposure)
    
    The key change: x_basis is ALWAYS [n_rays, n_materials] throughout the system.
    """
    def __init__(self, energies_keV, basis_materials, emission_op, filter_op, blur_op, 
                 interaction_op, detector_op, b=0.0, V=0.0, scatter_op=None, 
                 target_scatter=0.0, target_counts=1e6):
        super().__init__()
        self.register_buffer("energies_keV", torch.as_tensor(energies_keV, dtype=torch.float32))
        
        # Use BasisMaterials instead of raw Q matrix
        self.basis_materials = basis_materials
        
        # Create object attenuator from basis materials
        self.object_attenuator = ObjectAttenuator(basis_materials)
        
        self.emission_op = emission_op
        self.filter_op = filter_op
        self.blur_op = blur_op
        self.interaction_op = interaction_op
        self.detector_op = detector_op
        self.scatter_op = scatter_op or ZeroScatterModel()
        self.target_scatter = target_scatter
        self.register_buffer("b", torch.as_tensor(b, dtype=torch.float32))
        self.register_buffer("V", torch.as_tensor(V, dtype=torch.float32))
        self._is_calibrated = False
        
        # Lookup table for fast decomposition (computed lazily)
        self.lookup_table = None
        
        # Calibrate scatter model with system components
        self.scatter_op.calibrate_scatter(basis_materials, emission_op, filter_op)
        
        # Auto-calibrate fluence and detector on initialization
        self.calibrate_fluence(target_counts)
        self.calibrate_detector()
    
    def calibrate_fluence(self, target_counts=1e6):
        """Scale emission intensity so detected photons (q_detected) equals target_counts in air."""
        with torch.no_grad():
            # Create air measurement (zero attenuation)
            x_air = torch.zeros(1, self.basis_materials.n_materials, device=self.basis_materials.Q.device)
            
            # Compute detected photons with current emission intensity
            q_detected, _ = self.compute_q_stats(x_air)
            current_detected = q_detected.sum()
            
            # Calculate scale factor to achieve target detected photons
            scale_factor = target_counts / (current_detected + 1e-12)
            
            # Apply scale to emission intensity
            current_intensity = self.emission_op.get_intensity()
            self.emission_op.set_intensity(current_intensity * scale_factor)
            
            print(f"Calibrated fluence to {target_counts:.1e} detected photons (scale: {scale_factor.item():.4f})")

    def calibrate_detector(self, x_air=None):
        """Calibrate detector D so y_air = 1.0 in air for EACH channel."""
        with torch.no_grad():
            if x_air is None:
                x_air = torch.zeros(1, self.basis_materials.n_materials, device=self.basis_materials.Q.device)
            
            q_mu, _ = self.compute_q_stats(x_air)
            y_air = self.detector_op(q_mu).view(-1)
            
            scale = 1.0 / (y_air + 1e-12)
            self.detector_op.D.data *= scale.view(-1, 1)
            
            print(f"Calibrated detector: y_air={y_air.tolist()} -> scales={scale.tolist()}")
            if self.b.dim() > 0: self.b.data.zero_()
        
        self._is_calibrated = True
        return self

    @property
    def Q_basis(self):
        """Access to basis attenuation spectrum matrix for backward compatibility."""
        return self.basis_materials.Q

    def deconvolve_system_blur(self, y_blur_obs):
        """Perform system blur correction on measurements y_blur_obs."""
        return self.blur_op.reverse(y_blur_obs)

    def compute_q_stats(self, x_basis):
        """
        Compute mean and variance of the spectral intensity vector q.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals (ALWAYS [n_rays, n_materials])
        
        Returns:
            q_mean: [n_rays, n_exposures, n_energies] - mean spectral photon distribution
            q_var: [n_rays, n_exposures, n_energies] - variance (equals mean for Poisson)
        """
        # Validate input shape
        if x_basis.dim() != 2:
            raise ValueError(f"x_basis must be [n_rays, n_materials], got shape {x_basis.shape}")
        
        n_rays = x_basis.shape[0]
        n_materials = x_basis.shape[1]
        
        if n_materials != self.basis_materials.n_materials:
            raise ValueError(f"x_basis has {n_materials} basis components, expected {self.basis_materials.n_materials}")
        
        # Emission (can be multi-exposure): [n_rays, n_exposures, n_energies]
        # Pass a dummy tensor with correct shape to trigger multi-exposure logic
        dummy = torch.ones(n_rays, 1, len(self.energies_keV), device=x_basis.device)
        q_emission = self.emission_op(dummy)
        
        # Filter (applies fixed-length attenuation): [n_rays, n_exposures, n_energies]
        q_filtered = self.filter_op(q_emission)
        
        # Apply object attenuation using the new ObjectAttenuator
        # object_attenuator takes (q, x_basis) and returns attenuated q
        q_primary = self.object_attenuator(q_filtered, x_basis)
        
        # Add scatter before interaction (both primary and scatter interact with detector)
        scatter = self.scatter_op(x_basis, q_attenuated=q_primary)
        q_incident = q_primary + scatter
        
        # Detector interaction (absorbs photons with energy-dependent probability)
        q_detected_mean = self.interaction_op(q_incident)

        # Handle the zero-count case, we never truly expect zero counts so lets clamp to q_detected_mean=0.001
        q_detected_mean = torch.clamp(q_detected_mean, min=0.001)
        
        # Variance for Poisson statistics: variance equals mean
        q_detected_var = q_detected_mean.clone()
        
        return q_detected_mean, q_detected_var

    def forward_stats(self, x_basis, ray_chunk_size: Optional[int] = None):
        """Propagate q stats through detector D (blur handled separately).
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            ray_chunk_size: Optional chunk size for ray-by-ray processing.
                           If None, processes all rays at once.
        
        Returns:
            y_mean: [n_rays, n_channels] - mean detector signal
            y_var: [n_rays, n_channels] - variance of detector signal
        """
        n_rays = x_basis.shape[0]
        
        if ray_chunk_size is None or ray_chunk_size >= n_rays:
            q_detected_mean, _ = self.compute_q_stats(x_basis)
            y_mean, y_var = self.detector_op.compute_forward_stats(q_detected_mean)
        else:
            # Chunked processing
            y_mean_chunks = []
            y_var_chunks = []
            for start in range(0, n_rays, ray_chunk_size):
                end = min(start + ray_chunk_size, n_rays)
                x_chunk = x_basis[start:end]
                
                q_detected_chunk, _ = self.compute_q_stats(x_chunk)
                y_mean_chunk, y_var_chunk = self.detector_op.compute_forward_stats(q_detected_chunk)
                y_mean_chunks.append(y_mean_chunk)
                y_var_chunks.append(y_var_chunk)
                
                del q_detected_chunk
            
            y_mean = torch.cat(y_mean_chunks, dim=0)
            y_var = torch.cat(y_var_chunks, dim=0)
            del y_mean_chunks, y_var_chunks
        
        # Add bias and noise floor
        b = self.b.view(1, -1) if self.b.dim() > 0 else self.b
        V = self.V.view(1, -1) if self.V.dim() > 0 else self.V
        
        return y_mean + b, y_var + V

    def log_prob(self, y_obs, x_basis, ray_chunk_size: Optional[int] = None, apply_blur: bool = False, reduction: str = 'sum', fixed_covariance: bool = False):
        """Compute log probability of observations given basis coefficients.
        
        Args:
            y_obs: [n_rays, n_channels] - observations
            x_basis: [n_rays, n_materials] - basis line integrals
            ray_chunk_size: Optional chunk size for ray-by-ray processing.
            apply_blur: Whether to deconvolve system blur. Set to False by default.
            reduction: 'sum' (scalar result) or 'none' (tensor result [n_rays])
            fixed_covariance: If True, treats variance as constant for gradient (Weighted Least Squares).
                             This avoids bias at low signal/high noise floor where grad log(var) is non-zero.
        
        Returns:
            log_prob: scalar or tensor - log probability
        """
        # Deconvolve system blur if needed (blur modeled as post-processing)
        if apply_blur:
            y_obs_deconv = self.deconvolve_system_blur(y_obs)
        else:
            y_obs_deconv = y_obs

        y_mu, y_var = self.forward_stats(x_basis, ray_chunk_size=ray_chunk_size)

        if fixed_covariance:
            y_var = y_var.detach()

        diff = y_obs_deconv - y_mu
        precision = 1.0 / torch.clamp(y_var, min=1e-10)
        nll = 0.5 * (torch.log(2.0 * np.pi * torch.clamp(y_var, min=1e-10)) + (diff**2) * precision)
        
        # Collapse over channels
        if nll.dim() > 1:
            nll = nll.sum(dim=-1)
            
        if reduction == 'sum':
            return -nll.sum()
        else:
            return -nll

    def sample(self, x_basis, ray_chunk_size: Optional[int] = None, apply_blur: bool = True):
        """
        Sample measurements with Poisson noise.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            ray_chunk_size: Optional chunk size for ray-by-ray processing. 
                           If None, processes all rays at once. Use this to reduce memory usage.
            apply_blur: Whether to apply system blur. Set to False if blur will be applied
                       separately. Blur is modeled as a post-processing convolution applied
                       after the primary forward model (Poisson noise + detector integration).
            
        Returns:
            y_sample: [n_src, n_rays, n_channels] - sampled detector measurements
        """
        with torch.no_grad():
            n_rays = x_basis.shape[0]
            
            # If no chunking requested or chunk is larger than n_rays, process all at once
            if ray_chunk_size is None or ray_chunk_size >= n_rays:
                # Get spectral distribution
                q_mean, _ = self.compute_q_stats(x_basis)  # [n_rays, n_exposures, n_energies]
                
                # Sample Poisson noise in spectral domain
                q_sample = torch.poisson(torch.clamp(q_mean, min=0.0))
                
                # Integrate through detector
                y_raw = self.detector_op(q_sample)  # [n_rays, n_chan]
            else:
                # Chunked processing to reduce memory
                y_raw_chunks = []
                for start in range(0, n_rays, ray_chunk_size):
                    end = min(start + ray_chunk_size, n_rays)
                    x_chunk = x_basis[start:end]
                    
                    # Get spectral distribution for chunk
                    q_mean_chunk, _ = self.compute_q_stats(x_chunk)
                    
                    # Sample Poisson noise
                    q_sample_chunk = torch.poisson(torch.clamp(q_mean_chunk, min=0.0))
                    
                    # Integrate through detector
                    y_raw_chunk = self.detector_op(q_sample_chunk)  # [chunk_size, n_chan]
                    y_raw_chunks.append(y_raw_chunk)
                    
                    # Free intermediate tensors
                    del q_mean_chunk, q_sample_chunk
                
                y_raw = torch.cat(y_raw_chunks, dim=0)  # [n_rays, n_chan]
                del y_raw_chunks
            
            # Apply system blur as post-processing step (after Poisson noise and detector integration)
            if apply_blur:
                y_blurred = self.blur_op(y_raw.unsqueeze(0))  # [1, n_rays, n_chan]
            else:
                y_blurred = y_raw.unsqueeze(0)  # [1, n_rays, n_chan]
            
            # Add bias and readout noise
            b = self.b.view(1, 1, -1) if self.b.dim() > 0 else self.b
            V = self.V.view(1, 1, -1) if self.V.dim() > 0 else self.V
            
            # Readout noise (Gaussian)
            readout_noise = torch.randn_like(y_blurred) * torch.sqrt(torch.clamp(V, min=0.0))
            
            return y_blurred + b + readout_noise
    
    def sample_q(self, x_basis):
        """
        Sample spectral photon distribution q with Poisson noise.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            
        Returns:
            q_sample: [n_rays, n_exposures, n_energies] - sampled spectral distribution
        """
        with torch.no_grad():
            q_mean, _ = self.compute_q_stats(x_basis)
            return torch.poisson(torch.clamp(q_mean, min=0.0))

    def compute_q(self, x_basis):
        """
        Compute mean spectral distribution (no noise).
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            
        Returns:
            q: [n_rays, n_exposures, n_energies] - mean spectral photon distribution
        """
        q, _ = self.compute_q_stats(x_basis)
        return q
    
    def get_emission_spectrum(self, n_rays=1):
        """
        Get emission spectrum from source.
        
        Args:
            n_rays: Number of rays (default 1)
            
        Returns:
            q_emission: [n_rays, n_exposures, n_energies] - emission spectrum
        """
        dummy = torch.ones(n_rays, 1, len(self.energies_keV), device=self.energies_keV.device)
        return self.emission_op(dummy)
    
    def get_filtered_spectrum(self, n_rays=1):
        """
        Get emission spectrum after filtering.
        
        Args:
            n_rays: Number of rays (default 1)
            
        Returns:
            q_filtered: [n_rays, n_exposures, n_energies] - filtered spectrum
        """
        q_emission = self.get_emission_spectrum(n_rays)
        return self.filter_op(q_emission)
    
    def get_attenuated_spectrum(self, x_basis):
        """
        Get spectrum after object attenuation (before detector interaction).
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            
        Returns:
            q_primary: [n_rays, n_exposures, n_energies] - attenuated primary spectrum
        """
        n_rays = x_basis.shape[0]
        dummy = torch.ones(n_rays, 1, len(self.energies_keV), device=x_basis.device)
        q_emission = self.emission_op(dummy)
        q_filtered = self.filter_op(q_emission)
        q_primary = self.object_attenuator(q_filtered, x_basis)
        return q_primary
    
    def get_incident_spectrum(self, x_basis):
        """
        Get spectrum incident on detector (after object and scatter, before interaction).
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            
        Returns:
            q_incident: [n_rays, n_exposures, n_energies] - incident spectrum with scatter
        """
        q_primary = self.get_attenuated_spectrum(x_basis)
        scatter = self.scatter_op(x_basis, q_attenuated=q_primary)
        return q_primary + scatter

    def forward(self, x_basis):
        y_mu, _ = self.forward_stats(x_basis)
        return y_mu

    def decompose_basis(
        self,
        y_obs: torch.Tensor,
        x_init: Optional[torch.Tensor] = None,
        learning_rate: float = 0.9,
        max_iters: int = 50,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        callback: Optional[callable] = None,
        store_history: bool = True,
        use_lookup_table: bool = False,
        fixed_covariance: bool = False,
    ) -> torch.Tensor:
        """
        Decompose measurements into basis coefficients.
        
        This method can use either:
        1. Newton's method optimization (use_lookup_table=False, default)
        2. Pre-computed lookup table (use_lookup_table=True, requires self.lookup_table to be computed)
        
        The lookup table ONLY replaces the n_materials->n_channels optimization step.
        It provides fast decomposition by directly mapping measurements to basis coefficients.
        
        Args:
            y_obs: Observed measurements [n_rays, n_channels]
            x_init: Initial guess for basis coefficients [n_rays, n_materials].
                   Required if use_lookup_table=False, ignored if use_lookup_table=True.
            learning_rate: Newton step size (default: 0.9, only used if use_lookup_table=False)
            max_iters: Maximum iterations (default: 50, only used if use_lookup_table=False)
            x_min: Optional lower bound for basis coefficients
            x_max: Optional upper bound for basis coefficients
            callback: Optional callback(iteration, x_current, nll_current) for animations
            store_history: Whether to store optimization history (default: True)
            use_lookup_table: If True, use pre-computed lookup table (must call compute_lookup_table first).
                    If False, use Newton's method optimization (default).
            fixed_covariance: If True, uses Weighted Least Squares (fixed variance) for 
                             Newton optimization to avoid log-variance bias.
            
        Returns:
            x_final: Optimized basis coefficients [n_rays, n_materials]
            
        Example:
            >>> # Method 1: Newton's method
            >>> y_obs = torch.tensor([[0.0143, 0.0417]])  # [1, 2] dual-energy measurements
            >>> x_init = torch.tensor([[50.0, 50.0]])     # [1, 2] initial PE/CS guess
            >>> x_basis = xray_system.decompose_basis(y_obs, x_init, x_min=0, x_max=400)
            >>> 
            >>> # Method 2: Lookup table (faster)
            >>> xray_system.compute_lookup_table()  # Compute once
            >>> x_basis = xray_system.decompose_basis(y_obs, use_lookup_table=True)
        """
        # Use lookup table if requested
        if use_lookup_table:
            if self.lookup_table is None:
                raise ValueError("Lookup table not computed. Call compute_lookup_table() first.")
            return self.lookup_table(y_obs)
        
        # Otherwise, use Newton's method optimization
        if x_init is None:
            raise ValueError("x_init is required when use_lookup_table=False")
        
        # Create log_prob function that uses XraySystem's log_prob method
        # Note: NewtonOptimizer passes (x, y) but log_prob expects (y, x)
        def log_prob_fn(x, y):
            return self.log_prob(y, x, fixed_covariance=fixed_covariance)  
        
        # Initialize optimizer with log_prob
        optimizer = NewtonOptimizer(
            forward_fn=self.forward_stats,  # Still needed for compatibility
            y_obs=y_obs,
            x_init=x_init,
            learning_rate=learning_rate,
            max_iters=max_iters,
            x_min=x_min,
            x_max=x_max,
            device=self.energies_keV.device,
            use_log_prob=True,
            log_prob_fn=log_prob_fn,
        )
        
        # Run optimization
        x_final = optimizer.optimize(callback=callback, store_history=store_history)
        
        return x_final


    def compute_lookup_table(
        self,
        x_min_grid: float = 0.0,
        x_max_grid: float = 600.0,
        n_grid: int = 50,
        lut_res: int = 512,
        ray_batch_size: int = 10000,
        learning_rate: float = 0.8,
        hessian_reg: float = 0.01,
        iterations: int = 25,
        x_init_vals: Optional[List[float]] = None,
        use_pca: bool = False,
        verbose: bool = True,
    ) -> DifferentiableLookupTable:
        """
        Compute and store a differentiable lookup table for fast material decomposition.
        
        This method creates a lookup table mapping dual-energy measurements (y_obs)
        to material basis coefficients. 
        
        Args:
            x_min_grid: Minimum PE/CS value to explore for PCA (default: 0.0)
            x_max_grid: Maximum PE/CS value to explore for PCA (default: 600.0)
            n_grid: Resolution of the range-finding grid (default: 50)
            lut_res: Resolution of the final LUT [lut_res, lut_res] (default: 512)
            ray_batch_size: Chunks for Newton's method (default: 10000)
            learning_rate: Newton step size for LUT computation (default: 0.8)
            hessian_reg: Hessian regularization (default: 0.01)
            iterations: Newton iterations per grid point (default: 25)
            x_init_vals: Initial guess for [PE, CS] (default: [200.0, 200.0])
            use_pca: If True, use PCA to rotate the LUT coordinate system (default: True)
            verbose: If True, print progress (default: True)
            
        Returns:
            lookup_table: The computed DifferentiableLookupTable module.
        """
        device = self.energies_keV.device
        
        if verbose:
            mode_str = "PCA-Rotated" if use_pca else "Axis-Aligned"
            print(f"--- Computing {mode_str} LookupTable ({lut_res}x{lut_res}) ---")
            
        # 1. Coordinate range finding (PCA vs Axis-Aligned)
        with torch.no_grad():
            pe_vals = torch.linspace(x_min_grid, x_max_grid, n_grid, device=device)
            cs_vals = torch.linspace(x_min_grid, x_max_grid, n_grid, device=device)
            PE, CS = torch.meshgrid(pe_vals, cs_vals, indexing='ij')
            x_samples = torch.stack([PE.flatten(), CS.flatten()], dim=1)
            
            y_mean_samples, _ = self.forward_stats(x_samples)
            y_log_samples = -torch.log(torch.clamp(y_mean_samples, min=1e-12))
            
            if use_pca:
                # PCA mode: z-score transformation z = (y_log - mean) @ rotation / std
                offset = y_log_samples.mean(dim=0, keepdim=True)
                y_centered = y_log_samples - offset
                cov = (y_centered.T @ y_centered) / (y_centered.shape[0] - 1)
                evals, evecs = torch.linalg.eigh(cov)
                idx = evals.argsort(descending=True)
                rotation_matrix = evecs[:, idx]
                
                y_rotated = y_centered @ rotation_matrix
                # Determine bounds from data with buffer
                z_min = y_rotated.min(dim=0, keepdim=True)[0] - 0.5
                z_max = y_rotated.max(dim=0, keepdim=True)[0] + 0.5
                
                # We still want to center on the mean and scale by std for "z-score" meaning
                scale = torch.std(y_rotated, dim=0, keepdim=True)
                
                transformer = CoordinateTransformer(rotation_matrix, offset, scale)

                # 2. Define the LUT input grid in z-score space [z_min/std, z_max/std]
                # To maintain [-1, 1] relative to the grid itself, we store the actual range
                self.register_buffer("lut_z_min", z_min)
                self.register_buffer("lut_z_max", z_max)
                
                z0_grid_vals = torch.linspace(z_min[0,0].item(), z_max[0,0].item(), lut_res, device=device)
                z1_grid_vals = torch.linspace(z_min[0,1].item(), z_max[0,1].item(), lut_res, device=device)
                u, v = torch.meshgrid(z0_grid_vals, z1_grid_vals, indexing='ij')
                y_rotated_grid = torch.stack([u.flatten(), v.flatten()], dim=1)
                
                # y_log = y_rotated_grid @ rotation_matrix.T + offset
                y_log_grid = (y_rotated_grid @ rotation_matrix.T) + offset
            else:
                # Axis-Aligned mode: raw log-space, no offset or special scaling
                offset = torch.zeros((1, y_log_samples.shape[1]), device=device)
                rotation_matrix = torch.eye(y_log_samples.shape[1], device=device)
                scale = torch.ones((1, y_log_samples.shape[1]), device=device)
                
                transformer = CoordinateTransformer(rotation_matrix, offset, scale)

                # Use data-driven bounds for log-space as well
                log_min = y_log_samples.min(dim=0, keepdim=True)[0] - 0.5
                log_max = y_log_samples.max(dim=0, keepdim=True)[0] + 0.5
                self.register_buffer("lut_z_min", log_min)
                self.register_buffer("lut_z_max", log_max)

                l0_vals = torch.linspace(log_min[0,0].item(), log_max[0,0].item(), lut_res, device=device)
                l1_vals = torch.linspace(log_min[0,1].item(), log_max[0,1].item(), lut_res, device=device)
                u, v = torch.meshgrid(l0_vals, l1_vals, indexing='ij')
                y_log_grid = torch.stack([u.flatten(), v.flatten()], dim=1)

        # 3. Map grid points to physical measurements for solver
        y_obs_grid = torch.exp(-y_log_grid)
        
        # 3. Solve for material basis at each grid point
        default_init = [x_max_grid/2.0, x_max_grid/2.0]
        x_init_vals = x_init_vals or default_init
        x_lut_list = []
        
        for i in range(0, y_obs_grid.shape[0], ray_batch_size):
            batch_y_obs = y_obs_grid[i : i + ray_batch_size]
            
            lut_optimizer = NewtonOptimizer(
                forward_fn=self.forward_stats,
                y_obs=batch_y_obs,
                x_init=torch.tensor([x_init_vals] * batch_y_obs.shape[0], device=device, dtype=torch.float32),
                learning_rate=learning_rate,
                hessian_reg_factor=hessian_reg,
                max_iters=iterations,
                x_min=x_min_grid,
                x_max=x_max_grid,
            )
            x_batch = lut_optimizer.optimize()
            x_lut_list.append(x_batch.detach())
            
            if verbose and (i // ray_batch_size) % 5 == 0:
                print(f"    Processed {min(i + ray_batch_size, y_obs_grid.shape[0])}/{y_obs_grid.shape[0]}...")

        x_lut_flat = torch.cat(x_lut_list, dim=0)
        lut_table = x_lut_flat.view(lut_res, lut_res, 2)
        
        # 4. Create and store the module
        self.lookup_table = DifferentiableLookupTable(lut_table, transformer, self.lut_z_min, self.lut_z_max)
        
        if verbose:
            print("  LookupTable computation complete.")
            
        return self.lookup_table

    def build_decomposition_lookup_table(self, **kwargs):
        """Alias for compute_lookup_table for backward compatibility."""
        return self.compute_lookup_table(**kwargs)

    def decompose_with_lookup_table(self, y_obs, lookup_table=None):
        """Decompose measurements using a lookup table."""
        lt = lookup_table if lookup_table is not None else self.lookup_table
        if lt is None:
            raise ValueError("No lookup table available. Call compute_lookup_table first.")
        return lt(y_obs)

    def build_decomposition_lut(self, **kwargs):
        """Legacy alias for compute_lookup_table."""
        return self.compute_lookup_table(**kwargs)

    def decompose_with_lut(self, **kwargs):
        """Legacy alias for decompose_with_lookup_table."""
        return self.decompose_with_lookup_table(**kwargs)
