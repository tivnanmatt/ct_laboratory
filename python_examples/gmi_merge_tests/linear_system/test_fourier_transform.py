import torch
import pytest
import numpy as np
from ct_laboratory.linear_system import FourierTransform


class TestFourierTransform:
    """Test cases for the FourierTransform linear operator."""
    
    def test_instantiation(self):
        """Test that FourierTransform can be instantiated correctly."""
        ft = FourierTransform(dim=(-2, -1))
        assert isinstance(ft, FourierTransform)
        assert ft.dim == (-2, -1)
    
    def test_forward_operation(self):
        """Test the forward Fourier transform operation."""
        # Create a simple 2D test signal
        x = torch.randn(3, 4, 5, 6)  # batch, channels, height, width
        
        ft = FourierTransform(dim=(-2, -1))
        result = ft.forward(x)
        
        # Check that the output has the same shape
        assert result.shape == x.shape
        
        # Check that the result is complex
        assert torch.is_complex(result)
    
    def test_inverse_operation(self):
        """Test that forward followed by inverse gives the original signal."""
        x = torch.randn(2, 3, 4, 5)
        
        ft = FourierTransform(dim=(-2, -1))
        x_fft = ft.forward(x)
        x_reconstructed = ft.conjugate_transpose(x_fft)
        
        # Check that we get back the original signal (within numerical precision)
        assert torch.allclose(x, x_reconstructed.real, atol=1e-10)
        assert torch.allclose(torch.zeros_like(x), x_reconstructed.imag, atol=1e-10)
    
    def test_unitary_property(self):
        """Test that the Fourier transform preserves inner products (unitary property)."""
        x1 = torch.randn(2, 3, 4, 5)
        x2 = torch.randn(2, 3, 4, 5)
        
        ft = FourierTransform(dim=(-2, -1))
        
        # Original inner product
        inner_product_original = torch.sum(torch.conj(x1) * x2)
        
        # Inner product in Fourier domain
        x1_fft = ft.forward(x1)
        x2_fft = ft.forward(x2)
        inner_product_fourier = torch.sum(torch.conj(x1_fft) * x2_fft)
        
        # Should be equal (unitary property) - convert to complex for comparison
        inner_product_original_complex = torch.complex(inner_product_original, torch.tensor(0.0))
        assert torch.allclose(inner_product_original_complex, inner_product_fourier, atol=1e-10)
    
    def test_transpose_operation(self):
        """Test the transpose operation."""
        x = torch.randn(2, 3, 4, 5)
        
        ft = FourierTransform(dim=(-2, -1))
        x_transpose = ft.transpose(x)
        
        # Check that the output has the same shape
        assert x_transpose.shape == x.shape
        
        # Check that the result is complex
        assert torch.is_complex(x_transpose)
    
    def test_conjugate_operation(self):
        """Test the conjugate operation."""
        x = torch.randn(2, 3, 4, 5)
        
        ft = FourierTransform(dim=(-2, -1))
        x_conj = ft.conjugate(x)
        
        # Check that the output has the same shape
        assert x_conj.shape == x.shape
        
        # Check that the result is complex
        assert torch.is_complex(x_conj)
    
    def test_conjugate_transpose_operation(self):
        """Test the conjugate transpose operation."""
        x = torch.randn(2, 3, 4, 5)
        
        ft = FourierTransform(dim=(-2, -1))
        x_conj_transpose = ft.conjugate_transpose(x)
        
        # Check that the output has the same shape
        assert x_conj_transpose.shape == x.shape
        
        # Check that the result is complex
        assert torch.is_complex(x_conj_transpose)
    
    def test_inheritance(self):
        """Test that FourierTransform inherits from UnitaryLinearSystem."""
        ft = FourierTransform(dim=(-2, -1))
        
        # Check inheritance chain
        from ct_laboratory.linear_system import UnitaryLinearSystem, InvertibleLinearSystem, LinearSystem
        
        assert isinstance(ft, UnitaryLinearSystem)
        assert isinstance(ft, InvertibleLinearSystem)
        assert isinstance(ft, LinearSystem)
    
    def test_different_dimensions(self):
        """Test Fourier transform with different dimension specifications."""
        x = torch.randn(2, 3, 4, 5)
        
        # Test with single dimension
        ft1 = FourierTransform(dim=-1)
        result1 = ft1.forward(x)
        assert result1.shape == x.shape
        
        # Test with tuple of dimensions
        ft2 = FourierTransform(dim=(-2, -1))
        result2 = ft2.forward(x)
        assert result2.shape == x.shape
        
        # Test with positive dimensions
        ft3 = FourierTransform(dim=(2, 3))
        result3 = ft3.forward(x)
        assert result3.shape == x.shape
    
    def test_known_signal(self):
        """Test with a known signal to verify correct Fourier transform."""
        # Create a simple delta function
        x = torch.zeros(1, 1, 8, 8)
        x[0, 0, 4, 4] = 1.0  # Center pixel
        
        ft = FourierTransform(dim=(-2, -1))
        x_fft = ft.forward(x)
        
        # The Fourier transform of a delta function should be constant
        # Check that all values have the same magnitude
        magnitudes = torch.abs(x_fft)
        assert torch.allclose(magnitudes, magnitudes[0, 0, 0, 0], atol=1e-10)
    
    def test_batch_processing(self):
        """Test that the operator works correctly with batched inputs."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 8, 8)
        
        ft = FourierTransform(dim=(-2, -1))
        result = ft.forward(x)
        
        assert result.shape == x.shape
        assert result.shape[0] == batch_size 