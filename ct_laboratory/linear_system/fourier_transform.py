import torch
from .unitary import UnitaryLinearSystem


class FourierTransform(UnitaryLinearSystem):
    """
    A linear system that implements the N-dimensional Fourier transform.
    
    This class implements a N-dimensional Fourier transform that can be used in a PyTorch model.
    It assumes the central pixel in the image is at the center of the input tensor in all dimensions
    and returns the Fourier transform with the zero-frequency component in the center of the image.
    """
    
    def __init__(self, dim):
        """
        Initialize the Fourier transform operator.
        
        Parameters:
            dim: int or tuple of ints
                The dimensions along which to perform the Fourier transform.
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward Fourier transform.
        
        Parameters:
            x: torch.Tensor
                The input tensor to transform.
                
        Returns:
            torch.Tensor: The Fourier transform of the input.
        """
        x_ifftshift = torch.fft.ifftshift(x, dim=self.dim)
        x_fft = torch.fft.fftn(x_ifftshift, dim=self.dim, norm="ortho")
        x_fftshift = torch.fft.fftshift(x_fft, dim=self.dim)
        return x_fftshift
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the conjugate transpose (inverse Fourier transform).
        
        Parameters:
            y: torch.Tensor
                The input tensor to inverse transform.
                
        Returns:
            torch.Tensor: The inverse Fourier transform of the input.
        """
        y_ifftshift = torch.fft.ifftshift(y, dim=self.dim)
        y_ifft = torch.fft.ifftn(y_ifftshift, dim=self.dim, norm="ortho")
        y_fftshift = torch.fft.fftshift(y_ifft, dim=self.dim)
        return y_fftshift
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose operation.
        
        Parameters:
            y: torch.Tensor
                The input tensor.
                
        Returns:
            torch.Tensor: The transpose of the operator applied to the input.
        """
        return torch.conj(self.conjugate_transpose(torch.conj(y)))
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the conjugate operation.
        
        Parameters:
            x: torch.Tensor
                The input tensor.
                
        Returns:
            torch.Tensor: The conjugate of the input.
        """
        return torch.conj(self.forward(torch.conj(x)))

    def transpose_LinearSystem(self):
        """
        Returns the transpose linear system. For the Fourier transform, this is itself (unitary).
        """
        return FourierTransform(self.dim)

    def conjugate_LinearSystem(self):
        """
        Returns the conjugate linear system. For the Fourier transform, this is itself (unitary).
        """
        return FourierTransform(self.dim)

    def conjugate_transpose_LinearSystem(self):
        """
        Returns the conjugate transpose linear system. For the Fourier transform, this is itself (unitary).
        """
        return FourierTransform(self.dim)

    def inverse_LinearSystem(self):
        """
        Returns the inverse linear system. For the Fourier transform, this is itself (unitary).
        """
        return FourierTransform(self.dim) 