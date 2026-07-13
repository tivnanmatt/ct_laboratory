"""
Measurement simulator for CT slice imaging with downsampling + upsampling and structured noise.

This module implements a measurement simulator that:
1. Downsamples input images via 4x4 average pooling
2. Upsamples back via nearest neighbor interpolation (measurements same size as input)
3. Adds white noise in the low-resolution domain, which becomes structured noise in high-res

The noise covariance structure is: Σ = A @ (σ²I) @ A^T
where A is the composite down-then-up operator.
"""

import torch
import torch.nn.functional as F
from ..linalg.core import LinearSystem
from .gaussian import LinearSystemGaussianNoise


class DownsampleUpsampleOperator(LinearSystem):
    """
    Linear operator that downsamples via average pooling, then upsamples via nearest neighbor.
    
    This creates measurements of the same size as the input, but with low-frequency content.
    
    Args:
        downsample_factor: Factor by which to downsample (default: 4)
        mode: Upsampling mode, either 'nearest' or 'bilinear' (default: 'nearest')
    """
    
    def __init__(self, downsample_factor=4, mode='nearest'):
        super(DownsampleUpsampleOperator, self).__init__()
        self.downsample_factor = downsample_factor
        self.mode = mode
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply downsampling followed by upsampling.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, channels, height, width]
        """
        # Store original shape
        original_shape = x.shape
        
        # Downsample via average pooling
        x_down = F.avg_pool2d(x, kernel_size=self.downsample_factor, stride=self.downsample_factor)
        
        # Upsample via nearest neighbor (or bilinear)
        x_up = F.interpolate(x_down, size=(original_shape[2], original_shape[3]), 
                            mode=self.mode, align_corners=None if self.mode == 'nearest' else False)
        
        return x_up
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transpose operation: A^T where A is the composite operator.
        
        For the composite operator A = U @ D (upsample @ downsample):
        A^T = D^T @ U^T
        
        Since both operations are linear:
        - D^T (transpose of average pooling) is sum-pooling (accumulation)
        - U^T (transpose of nearest neighbor) is downsampling with summation of replicated values
        
        Args:
            y: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor of shape [batch_size, channels, height, width]
        """
        # For nearest neighbor upsampling followed by average pooling,
        # the transpose is: downsample by summation, then upsample by replication
        # But since we want the same output shape, we use autograd
        
        # Use autograd to compute the transpose
        if not hasattr(self, '_input_shape') or self._input_shape != y.shape:
            self._input_shape = y.shape
        
        # Create a dummy input and use autograd
        x_dummy = torch.zeros(y.shape, dtype=y.dtype, device=y.device, requires_grad=True)
        output = self.forward(x_dummy)
        
        # Compute transpose using backprop
        output.backward(y, create_graph=True)
        result = x_dummy.grad
        
        return result


class CTSliceMeasurementSimulator(LinearSystemGaussianNoise):
    """
    Measurement simulator for CT slices with structured noise.
    
    This implements the measurement model:
        y = A(x) + η
    
    where:
        - A is a downsampling + upsampling operator (low-pass filter)
        - η ~ N(0, Σ) with Σ = A @ (σ²I) @ A^T (structured noise)
    
    The noise is white in the low-resolution domain but becomes structured 
    (correlated, low-frequency) in the high-resolution domain.
    
    Args:
        downsample_factor: Factor by which to downsample (default: 4)
        noise_variance: Variance of white noise in low-res domain (default: 0.01)
        upsampling_mode: Upsampling mode, 'nearest' or 'bilinear' (default: 'nearest')
    """
    
    def __init__(self, downsample_factor=4, noise_variance=0.01, upsampling_mode='nearest'):
        # Create the downsampling-upsampling linear operator
        linear_system = DownsampleUpsampleOperator(
            downsample_factor=downsample_factor,
            mode=upsampling_mode
        )
        
        # Create noise covariance: Σ = A @ (σ²I) @ A^T
        # This is implicitly represented by applying A to white noise
        noise_covariance = StructuredNoiseCovariance(
            linear_system=linear_system,
            noise_variance=noise_variance
        )
        
        # Initialize parent class
        super(CTSliceMeasurementSimulator, self).__init__(
            linear_system=linear_system,
            noise_covariance=noise_covariance
        )
        
        self.downsample_factor = downsample_factor
        self.noise_variance = noise_variance
        self.upsampling_mode = upsampling_mode
    
    def sample(self, batch_size, x):
        """
        Generate measurements from input images.
        
        Args:
            batch_size: Number of measurement samples to generate per image
            x: Input images of shape [image_batch_size, channels, height, width]
            
        Returns:
            Measurements of shape [batch_size, image_batch_size, channels, height, width]
            or [image_batch_size, channels, height, width] if batch_size == 1
        """
        image_batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        
        # Apply the linear system (downsample + upsample)
        mean = self.linear_system(x)  # Shape: [image_batch_size, channels, height, width]
        
        # Generate structured noise samples
        measurements = []
        for _ in range(batch_size):
            # Generate white noise in low-res domain
            low_res_height = height // self.downsample_factor
            low_res_width = width // self.downsample_factor
            
            white_noise = torch.randn(
                image_batch_size, channels, low_res_height, low_res_width,
                device=x.device, dtype=x.dtype
            ) * torch.sqrt(torch.tensor(self.noise_variance, device=x.device, dtype=x.dtype))
            
            # Upsample noise to high-res (this applies A to white noise in low-res)
            structured_noise = F.interpolate(
                white_noise,
                size=(height, width),
                mode=self.upsampling_mode,
                align_corners=None if self.upsampling_mode == 'nearest' else False
            )
            
            # Add noise to mean
            measurement = mean + structured_noise
            measurements.append(measurement)
        
        # Stack measurements
        if batch_size == 1:
            return measurements[0]  # Shape: [image_batch_size, channels, height, width]
        else:
            return torch.stack(measurements)  # Shape: [batch_size, image_batch_size, channels, height, width]


class StructuredNoiseCovariance(LinearSystem):
    """
    Represents noise covariance Σ = A @ (σ²I) @ A^T for structured noise.
    
    This is used to represent the noise covariance when white noise in a low-resolution
    domain is mapped to a high-resolution domain via a linear operator A.
    
    Args:
        linear_system: The linear operator A (e.g., upsampling)
        noise_variance: Variance σ² of white noise in low-res domain
    """
    
    def __init__(self, linear_system, noise_variance):
        super(StructuredNoiseCovariance, self).__init__()
        self.linear_system = linear_system
        self.noise_variance = noise_variance
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply covariance: Σ @ x = A @ (σ²I) @ A^T @ x
        
        Args:
            x: Input tensor
            
        Returns:
            Σ @ x
        """
        # First apply A^T
        x_transposed = self.linear_system.transpose(x)
        
        # Scale by noise variance (σ²I)
        x_scaled = x_transposed * self.noise_variance
        
        # Apply A
        result = self.linear_system.forward(x_scaled)
        
        return result
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transpose of covariance: Σ^T = (A @ (σ²I) @ A^T)^T = A @ (σ²I) @ A^T = Σ
        
        The covariance is symmetric, so transpose equals forward.
        """
        return self.forward(y)
    
    def sqrt_LinearSystem(self):
        """
        Return square root of covariance: Σ^(1/2) = A @ (σI)
        
        This can be used for generating samples via: Σ^(1/2) @ z where z ~ N(0, I)
        """
        return ScaledLinearSystem(self.linear_system, torch.sqrt(torch.tensor(self.noise_variance)))
    
    def inv_LinearSystem(self):
        """
        Inverse of covariance: Σ^(-1) = (A @ (σ²I) @ A^T)^(-1)
        
        This is complex for general operators. For practical purposes, 
        we can use pseudoinverse or conjugate gradient methods.
        """
        raise NotImplementedError(
            "Inverse of structured noise covariance is not directly implemented. "
            "Use pseudoinverse() method instead."
        )


class ScaledLinearSystem(LinearSystem):
    """
    Represents a linear system scaled by a scalar: S = c * A
    
    Args:
        linear_system: The linear operator A
        scale: The scalar c
    """
    
    def __init__(self, linear_system, scale):
        super(ScaledLinearSystem, self).__init__()
        self.linear_system = linear_system
        self.scale = scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply scaled operator: (c * A) @ x"""
        return self.scale * self.linear_system.forward(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """Transpose: (c * A)^T = c * A^T"""
        return self.scale * self.linear_system.transpose(y)
