import torch
import numpy as np

class InvertibleFourierGaussianFilter(torch.nn.Module):
    """FFT-based Gaussian blur with compact spatial kernel.
    
    Boundary conditions:
      - Column (X) direction: circular (FFT's natural periodic convolution)
      - Row    (Y) direction: reflection (via explicit reflect-padding before FFT)
    
    The filter is built from a finite-support spatial-domain Gaussian kernel
    (truncated at 4σ), then embedded into the FFT grid and transformed.
    This avoids wrap-around ringing that occurs when using the analytical
    infinite-extent frequency-domain Gaussian.
    """
    def __init__(self, n_rotations, n_frame, n_modules_per_source, nx, ny, fwhm_mm=1.0, spacing_x=1.0, spacing_y=1.0):
        super().__init__()
        self.n_rotations = n_rotations
        self.n_frame = n_frame
        self.n_modules_per_source = n_modules_per_source
        self.nx = nx
        self.ny = ny
        
        # sigma = FWHM / (2 * sqrt(2 * ln(2)))
        sigma_mm = fwhm_mm / 2.35482
        
        # Sigma in pixel units for each direction
        sigma_px_x = sigma_mm / spacing_x if spacing_x > 0 else 0
        sigma_px_y = sigma_mm / spacing_y if spacing_y > 0 else 0
        
        # Compact kernel half-width (truncate at 4*sigma, minimum 1 pixel)
        kernel_half_x = max(1, int(np.ceil(4 * sigma_px_x)))
        kernel_half_y = max(1, int(np.ceil(4 * sigma_px_y)))
        kernel_size_x = 2 * kernel_half_x + 1
        kernel_size_y = 2 * kernel_half_y + 1
        
        self.W = n_modules_per_source * nx
        self.H = ny
        
        # Padding: FULL size reflection in BOTH X and Y directions
        # This means we pad by the full image dimension on each side (triple total size)
        # Clamp to (dimension - 1) due to PyTorch's reflect padding constraint
        self.pad_x = min(self.W - 1, self.W) if self.W > 1 else 0
        self.pad_y = min(self.H - 1, self.H) if self.H > 1 else 0
        
        self.W_fft = self.W + 2 * self.pad_x
        self.H_fft = self.H + 2 * self.pad_y
        
        # --- Build compact spatial-domain Gaussian kernel ---
        kx_coords = torch.arange(-kernel_half_x, kernel_half_x + 1, dtype=torch.float32) * spacing_x  # mm
        ky_coords = torch.arange(-kernel_half_y, kernel_half_y + 1, dtype=torch.float32) * spacing_y  # mm
        KY_k, KX_k = torch.meshgrid(ky_coords, kx_coords, indexing='ij')
        
        if sigma_mm > 1e-12:
            kernel_spatial = torch.exp(-(KX_k**2 + KY_k**2) / (2 * sigma_mm**2))
        else:
            kernel_spatial = torch.zeros(kernel_size_y, kernel_size_x)
            kernel_spatial[kernel_half_y, kernel_half_x] = 1.0
        kernel_spatial = kernel_spatial / kernel_spatial.sum()  # normalise to unity
        
        # --- Embed kernel in FFT grid (center at origin, negative offsets wrap) ---
        kernel_fft_input = torch.zeros(self.H_fft, self.W_fft)
        # Clip kernel to fit in FFT grid (needed when dimensions are very small, e.g., dummy init)
        ksize_y_clip = min(kernel_size_y, self.H_fft)
        ksize_x_clip = min(kernel_size_x, self.W_fft)
        kernel_fft_input[:ksize_y_clip, :ksize_x_clip] = kernel_spatial[:ksize_y_clip, :ksize_x_clip]
        # Roll so that kernel centre sits at index (0, 0)
        kernel_fft_input = torch.roll(kernel_fft_input,
                                       shifts=(-min(kernel_half_y, self.H_fft-1), -min(kernel_half_x, self.W_fft-1)),
                                       dims=(0, 1))
        
        # --- Frequency-domain filter (DFT of the compact kernel) ---
        # For a real, symmetric kernel the DFT is real; take .real for safety.
        filter_2d = torch.fft.fft2(kernel_fft_input).real
        
        # Wiener-regularised inverse: H / (H² + ε)
        eps = 1e-3
        inv_filter_2d = filter_2d / (filter_2d**2 + eps)
        
        self.register_buffer("filter_2d", filter_2d)
        self.register_buffer("inv_filter_2d", inv_filter_2d)
        self.fwhm_mm = fwhm_mm
        
        # Store the compact spatial kernel for conv2d-based variance propagation
        self.register_buffer("kernel", kernel_spatial.view(1, 1, kernel_size_y, kernel_size_x))

    def _apply_filter(self, x, filter_weights):
        # x: [n_src, n_rays, n_out]
        # Projector ray memory layout per view: [n_modules_per_source, nx, ny]
        # We unflatten to a 2D detector image [H=ny, W=n_mod*nx] before filtering.
        n_src, n_rays, n_out = x.shape
        view_size = self.n_modules_per_source * self.nx * self.ny  # rays per view
        if n_rays % view_size != 0:
            return x
        n_views = n_rays // view_size
        
        # 1. Unflatten rays → [n_src, n_views, n_mod, nx, ny, n_out]
        x_mod = x.view(n_src, n_views, self.n_modules_per_source, self.nx, self.ny, n_out)
        # 2. Permute to 2D image [n_src, n_views, n_out, ny, n_mod, nx]
        #    then merge → [batch, H=ny, W=n_mod*nx]
        x_image = x_mod.permute(0, 1, 5, 4, 2, 3).reshape(-1, self.H, self.W)
        
        # 3. Padding: reflect in BOTH Y (rows) and X (columns) by FULL image size
        #    This creates triple size: (3*H, 3*W)
        if self.pad_x > 0 or self.pad_y > 0:
            x_padded = torch.nn.functional.pad(
                x_image, (self.pad_x, self.pad_x, self.pad_y, self.pad_y), mode='reflect')
        else:
            x_padded = x_image
        
        # 4. FFT convolution
        X_freq = torch.fft.fft2(x_padded)
        Y_freq = X_freq * filter_weights.unsqueeze(0)
        y_padded = torch.fft.ifft2(Y_freq).real
        
        # 5. Crop both Y and X padding back to original height and width
        if self.pad_y > 0 or self.pad_x > 0:
            y_image = y_padded[:, self.pad_y:self.pad_y + self.H, self.pad_x:self.pad_x + self.W]
        else:
            y_image = y_padded
        
        # 6. Reverse reshape → [n_src, n_rays, n_out]
        y_mod = y_image.view(n_src, n_views, n_out, self.ny, self.n_modules_per_source, self.nx)
        y_out = y_mod.permute(0, 1, 4, 5, 3, 2).reshape(n_src, n_rays, n_out)
        return y_out

    def forward(self, x):
        if self.fwhm_mm == 0: return x
        return self._apply_filter(x, self.filter_2d)

    def reverse(self, x):
        if self.fwhm_mm == 0: return x
        return self._apply_filter(x, self.inv_filter_2d)

class GaussianProjectionBlur(torch.nn.Module):
    def __init__(self, n_rotations, n_frame, n_modules_per_source, nx, ny, fwhm_mm=1.0, spacing_x=1.0, spacing_y=1.0):
        super().__init__()
        self.n_rotations = n_rotations
        self.n_frame = n_frame
        self.n_modules_per_source = n_modules_per_source
        self.nx = nx
        self.ny = ny
        sigma_x = (fwhm_mm / 2.35482) / spacing_x
        sigma_y = (fwhm_mm / 2.35482) / spacing_y
        ksize = 11
        x = torch.linspace(-(ksize // 2), ksize // 2, ksize, dtype=torch.float32)
        y = torch.linspace(-(ksize // 2), ksize // 2, ksize, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
        kernel = kernel / (kernel.sum() + 1e-12)
        self.register_buffer("kernel", kernel.view(1, 1, ksize, ksize))

    def forward(self, x):
        n_src, n_rays, n_out = x.shape
        view_size = self.n_modules_per_source * self.nx * self.ny
        if n_rays % view_size != 0: return x
        n_views = n_rays // view_size
        x_reshaped = x.transpose(1, 2).view(n_src * n_out * n_views, self.n_modules_per_source, self.nx, self.ny)
        x_image = x_reshaped.permute(0, 3, 1, 2).reshape(n_src * n_out * n_views, 1, self.ny, self.n_modules_per_source * self.nx)
        pad = (self.kernel.shape[-1] - 1) // 2
        if self.ny == 1:
            x_padded = torch.nn.functional.pad(x_image, (pad, pad, 0, 0), mode='reflect')
            x_padded = torch.nn.functional.pad(x_padded, (0, 0, pad, pad), mode='replicate')
        else:
            x_padded = torch.nn.functional.pad(x_image, (pad, pad, pad, pad), mode='reflect')
        x_blurred = torch.nn.functional.conv2d(x_padded, self.kernel)
        x_out = x_blurred.view(n_src * n_out * n_views, self.ny, self.n_modules_per_source, self.nx)
        x_out = x_out.permute(0, 2, 3, 1).reshape(n_src, n_out, n_rays).transpose(1, 2)
        return x_out

class IdentityBlurOperator(torch.nn.Module):
    def forward(self, x): return x
    def reverse(self, x): return x
