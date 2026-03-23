import torch
import torch.nn.functional as F
import numpy as np

class InvertibleFourierGaussianFilter(torch.nn.Module):
    """FFT-based Gaussian blur with compact spatial kernel.
    
    Boundary conditions:
      - Column (X) direction: circular (FFT's natural periodic convolution)
      - Row    (Y) direction: reflection (via explicit reflect-padding before FFT)
    """
    def __init__(self, n_rotations, n_frame, n_modules_per_source, nx, ny, fwhm_mm=1.0, spacing_x=1.0, spacing_y=1.0):
        super().__init__()
        # sigma = FWHM / (2.35482)
        sigma_mm = fwhm_mm / 2.35482
        sigma_px_x = sigma_mm / spacing_x if spacing_x > 0 else 0
        sigma_px_y = sigma_mm / spacing_y if spacing_y > 0 else 0
        
        kernel_half_x = max(1, int(np.ceil(4 * sigma_px_x)))
        kernel_half_y = max(1, int(np.ceil(4 * sigma_px_y)))
        
        self.W = n_modules_per_source * nx
        self.H = ny
        self.pad_y = min(kernel_half_y, self.H - 1)
        self.W_fft = self.W
        self.H_fft = self.H + 2 * self.pad_y
        
        kx_coords = torch.arange(-kernel_half_x, kernel_half_x + 1, dtype=torch.float32) * spacing_x
        ky_coords = torch.arange(-kernel_half_y, kernel_half_y + 1, dtype=torch.float32) * spacing_y
        KY_k, KX_k = torch.meshgrid(ky_coords, kx_coords, indexing='ij')
        
        if sigma_mm > 1e-12:
            kernel_spatial = torch.exp(-(KX_k**2 + KY_k**2) / (2 * sigma_mm**2))
        else:
            kernel_spatial = torch.zeros(2*kernel_half_y+1, 2*kernel_half_x+1)
            kernel_spatial[kernel_half_y, kernel_half_x] = 1.0
        kernel_spatial /= kernel_spatial.sum()
        
        kernel_fft_input = torch.zeros(self.H_fft, self.W_fft)
        y_center, x_center = self.H_fft // 2, self.W_fft // 2
        
        # Place kernel and shift to origin for FFT convolution
        kh, kw = kernel_spatial.shape
        y_slice = slice(0, kh)
        x_slice = slice(0, kw)
        kernel_fft_input[0:kh, 0:kw] = kernel_spatial
        kernel_fft_input = torch.roll(kernel_fft_input, shifts=(-kernel_half_y, -kernel_half_x), dims=(0, 1))
        
        transfer_function = torch.fft.rfft2(kernel_fft_input)
        self.register_buffer("transfer_function", transfer_function)

    def forward(self, x):
        # x: [Batch, H, W]
        # Pad Y with reflection
        x_padded = F.pad(x, (0, 0, self.pad_y, self.pad_y), mode='reflect')
        x_fft = torch.fft.rfft2(x_padded)
        y_fft = x_fft * self.transfer_function
        y_padded = torch.fft.irfft2(y_fft, s=(self.H_fft, self.W_fft))
        # Crop back to original size
        return y_padded[:, self.pad_y : self.pad_y + self.H, :]

    def reverse(self, y, eps=1e-3):
        # Deconvolution (Wiener-like or simple inverse with epsilon)
        y_padded = F.pad(y, (0, 0, self.pad_y, self.pad_y), mode='reflect')
        y_fft = torch.fft.rfft2(y_padded)
        inv_tf = torch.conj(self.transfer_function) / (torch.abs(self.transfer_function)**2 + eps)
        x_fft = y_fft * inv_tf
        x_padded = torch.fft.irfft2(x_fft, s=(self.H_fft, self.W_fft))
        return x_padded[:, self.pad_y : self.pad_y + self.H, :]
