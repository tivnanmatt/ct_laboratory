import torch
from .fourier_linear_operator import FourierFilter
from .fourier_transform import FourierTransform


class FourierConvolution(FourierFilter):
    """
    Implements a convolution operator using the Fourier domain.
    
    This class takes a kernel, computes its Fourier transform to create a filter,
    and then applies that filter in the Fourier domain for efficient convolution.
    """
    
    def __init__(self, kernel: torch.Tensor, dim):
        """
        Initialize the Fourier convolution operator.
        
        Parameters:
            kernel: torch.Tensor
                The convolution kernel to apply.
            dim: int or tuple of ints
                The dimensions along which to perform the Fourier transform.
        """
        # Compute the Fourier transform of the kernel to create the filter
        fourier_transform = FourierTransform(dim=dim)
        filter = fourier_transform.forward(kernel)
        
        # Initialize the parent FourierFilter with the computed filter
        super().__init__(filter, dim)
        self.kernel = kernel 

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # Use the parent class's inverse
        return super().inverse(y) 