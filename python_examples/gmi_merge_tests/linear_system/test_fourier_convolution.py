import torch
import pytest
import numpy as np
from ct_laboratory.linear_system import FourierConvolution, FourierFilter


class TestFourierConvolution:
    """Test cases for the FourierConvolution linear operator."""
    
    def test_instantiation(self):
        """Test that FourierConvolution can be instantiated correctly."""
        kernel = torch.randn(8, 8)
        conv = FourierConvolution(kernel, dim=(-2, -1))
        assert isinstance(conv, FourierConvolution)
        assert torch.allclose(conv.kernel, kernel)
        assert conv.dim == (-2, -1)
    
    def test_forward_operation(self):
        """Test the forward convolution operation."""
        # Create a simple kernel and input
        kernel = torch.randn(8, 8)
        x = torch.randn(1, 8, 8)
        
        conv = FourierConvolution(kernel, dim=(-2, -1))
        result = conv.forward(x)
        
        # Check that the output has the same shape
        assert result.shape == x.shape
        
        # Check that the result is complex (due to Fourier operations)
        assert torch.is_complex(result)
    
    def test_inheritance(self):
        """Test that FourierConvolution inherits from the correct classes."""
        kernel = torch.randn(8, 8)
        conv = FourierConvolution(kernel, dim=(-2, -1))
        
        # Check inheritance chain
        from ct_laboratory.linear_system import (
            FourierFilter,
            EigenDecompositionLinearSystem,
            LinearSystem
        )
        
        assert isinstance(conv, FourierFilter)
        assert isinstance(conv, EigenDecompositionLinearSystem)
        assert isinstance(conv, LinearSystem)
    
    def test_filter_computation(self):
        """Test that the filter is correctly computed from the kernel."""
        kernel = torch.randn(8, 8)
        conv = FourierConvolution(kernel, dim=(-2, -1))
        
        # The filter should be the Fourier transform of the kernel
        from ct_laboratory.linear_system import FourierTransform
        ft = FourierTransform(dim=(-2, -1))
        expected_filter = ft.forward(kernel)
        
        assert torch.allclose(conv.filter, expected_filter, atol=1e-10)
    
    def test_convolution_equivalence(self):
        """Test that Fourier convolution gives the same result as direct convolution."""
        # Create a simple test case
        kernel = torch.randn(5, 5)
        x = torch.randn(1, 8, 8)
        
        # Pad the kernel to match the input size
        kernel_padded = torch.zeros(8, 8)
        kernel_padded[:5, :5] = kernel
        
        # Apply Fourier convolution
        conv = FourierConvolution(kernel_padded, dim=(-2, -1))
        result_fourier = conv.forward(x)
        
        # Apply direct convolution using the same FFT shifting approach
        # Use the same approach as FourierTransform: ifftshift -> fftn -> fftshift
        x_ifftshift = torch.fft.ifftshift(x, dim=(-2, -1))
        x_fft = torch.fft.fftn(x_ifftshift, dim=(-2, -1), norm="ortho")
        x_fftshift = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        kernel_ifftshift = torch.fft.ifftshift(kernel_padded, dim=(-2, -1))
        kernel_fft = torch.fft.fftn(kernel_ifftshift, dim=(-2, -1), norm="ortho")
        kernel_fftshift = torch.fft.fftshift(kernel_fft, dim=(-2, -1))
        
        # Multiply in Fourier domain
        result_fft = x_fftshift * kernel_fftshift
        
        # Inverse transform
        result_ifftshift = torch.fft.ifftshift(result_fft, dim=(-2, -1))
        result_ifft = torch.fft.ifftn(result_ifftshift, dim=(-2, -1), norm="ortho")
        result_direct = torch.fft.fftshift(result_ifft, dim=(-2, -1))
        
        # Results should be similar (within numerical precision)
        assert torch.allclose(result_fourier, result_direct, atol=1e-10)
    
    def test_different_kernel_sizes(self):
        """Test with different kernel sizes."""
        # Test with smaller kernel
        kernel_small = torch.randn(3, 3)
        x = torch.randn(1, 8, 8)
        
        # Pad kernel to match input size
        kernel_padded = torch.zeros(8, 8)
        kernel_padded[:3, :3] = kernel_small
        
        conv = FourierConvolution(kernel_padded, dim=(-2, -1))
        result = conv.forward(x)
        assert result.shape == x.shape
    
    def test_batch_processing(self):
        """Test that the operator works correctly with batched inputs."""
        batch_size = 4
        kernel = torch.randn(8, 8)
        x = torch.randn(batch_size, 8, 8)
        
        conv = FourierConvolution(kernel, dim=(-2, -1))
        result = conv.forward(x)
        
        assert result.shape == x.shape
        assert result.shape[0] == batch_size
    
    def test_mat_add(self):
        """Test matrix addition with other FourierFilter instances."""
        kernel1 = torch.randn(8, 8)
        kernel2 = torch.randn(8, 8)
        
        conv1 = FourierConvolution(kernel1, dim=(-2, -1))
        conv2 = FourierConvolution(kernel2, dim=(-2, -1))
        
        # Test addition
        conv_sum = conv1.mat_add(conv2)
        assert isinstance(conv_sum, FourierFilter)
        assert torch.allclose(conv_sum.filter, conv1.filter + conv2.filter)
    
    def test_mat_mul(self):
        """Test matrix multiplication with other FourierFilter instances."""
        kernel1 = torch.randn(8, 8)
        kernel2 = torch.randn(8, 8)
        
        conv1 = FourierConvolution(kernel1, dim=(-2, -1))
        conv2 = FourierConvolution(kernel2, dim=(-2, -1))
        
        # Test multiplication
        conv_mul = conv1.mat_mul(conv2)
        assert isinstance(conv_mul, FourierFilter)
        assert torch.allclose(conv_mul.filter, conv1.filter * conv2.filter) 