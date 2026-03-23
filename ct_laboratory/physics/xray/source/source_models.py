import torch
import numpy as np

class UniformEmissionOperator(torch.nn.Module):
    """
    A basic emission operator that provides a uniform energy spectrum.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, I0, target_counts=1e6):
        super().__init__()
        I0_tensor = torch.as_tensor(I0, dtype=torch.float32)
        # We want to represent a single exposure by default.
        # I0 shape is [E]
        if I0_tensor.dim() == 1:
            I0_tensor = I0_tensor.view(1, 1, -1) # [1, 1, E]
        elif I0_tensor.dim() == 2:
            I0_tensor = I0_tensor.unsqueeze(0) # [1, N, E]
        self.register_buffer("I0", I0_tensor)
        self.target_counts = target_counts
    
    def get_intensity(self):
        """Get total photon intensity (sum over energies for each exposure)."""
        return self.I0.sum(dim=-1).view(-1)
    
    def set_intensity(self, intensity):
        """Set total photon intensity by scaling I0."""
        with torch.no_grad():
            current_intensity = self.get_intensity()
            scale = intensity / (current_intensity + 1e-12)
            self.I0.data *= scale.view(1, -1, 1)

    def forward(self, x):
        return self.I0 * x
    
    def get_spectrum(self, n_rays=1):
        """
        Get the emission spectrum without requiring a dummy tensor.
        
        Args:
            n_rays: Number of rays (default 1)
            
        Returns:
            Emission spectrum [n_rays, n_exposures, n_energies]
        """
        return self.I0.expand(n_rays, -1, -1)
    
    def sample_spectrum(self, n_rays=1):
        """
        Sample emission spectrum with Poisson noise.
        
        Args:
            n_rays: Number of rays (default 1)
            
        Returns:
            Poisson-sampled spectrum [n_rays, n_exposures, n_energies]
        """
        q_mean = self.get_spectrum(n_rays)
        return torch.poisson(torch.clamp(q_mean, min=0.0))
    
    def sample(self, x):
        """
        Sample emission spectrum with Poisson noise.
        
        Args:
            x: Input tensor [n_rays, ...]
            
        Returns:
            Poisson-sampled spectrum
        """
        q_mean = self.forward(x)
        return torch.poisson(torch.clamp(q_mean, min=0.0))

class XraySource(UniformEmissionOperator):
    """
    A base X-ray source that provides an emission spectrum.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    pass

class SpekpyXraySource(UniformEmissionOperator):
    """
    X-ray source that generates spectrum using spekpy.
    
    Args:
        kvp: Peak kilovoltage (e.g., 60, 80, 120, 140 kVp)
        dk: Energy bin width in keV (default 1.0)
        n_bins: Number of energy bins
        target_counts: Target photon count for calibration (default 1e6)
        
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, kvp, dk=1.0, n_bins=150, target_counts=1e6):
        try:
            import spekpy
        except ImportError:
            raise ImportError("spekpy is required for SpekpyXraySource. Install with: pip install spekpy")
        
        # Generate energy axis
        energies_keV = np.arange(dk/2, n_bins * dk, dk)
        
        # Generate spectrum using spekpy
        s = spekpy.Spek(kvp=kvp, dk=dk)
        k, sp = s.get_spectrum()
        
        # Map spectrum to energy bins
        spec_emission = np.zeros(n_bins)
        for i, val in enumerate(k):
            idx = np.argmin(np.abs(energies_keV - val))
            if idx < n_bins and val <= kvp:
                spec_emission[idx] = sp[i]
        
        # Initialize parent class first
        super().__init__(torch.from_numpy(spec_emission).float(), target_counts=target_counts)
        
        # Store energies as buffer for easy access
        self.register_buffer("energies_keV", torch.from_numpy(energies_keV).float())
        self.kvp = kvp
        self.dk = dk

class FilteredXraySource(torch.nn.Module):
    """
    An X-ray source that includes filtration/attenuation.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, source, attenuator):
        super().__init__()
        self.source = source
        self.attenuator = attenuator
    
    def get_intensity(self):
        """Get intensity from underlying source."""
        return self.source.get_intensity()
    
    def set_intensity(self, intensity):
        """Set intensity on underlying source."""
        self.source.set_intensity(intensity)

    def forward(self, x):
        # source(x) -> [n_rays, n_exposures, n_energies]
        q0 = self.source(x)
        return self.attenuator(q0)

class AluminiumFilteredSource(FilteredXraySource):
    """
    An X-ray source filtered by a specific thickness of Aluminum.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, energies_ev, I0, al_thickness_mm):
        from ct_laboratory.physics.xray.attenuation.operators import UniformAluminumFilter
        source = XraySource(I0)
        attenuator = UniformAluminumFilter(energies_ev, al_thickness_mm)
        super().__init__(source, attenuator)

class DualExposureXraySource(torch.nn.Module):
    """
    An X-ray source that generates two distinct exposures (e.g., Low and High kVp).
    
    Output Shape: [n_rays, n_exposures=2, n_energies]
    """
    def __init__(self, source_low, source_high):
        super().__init__()
        self.source_low = source_low
        self.source_high = source_high
    
    def get_intensity(self):
        """Get intensity for both exposures."""
        i_low = self.source_low.get_intensity()
        i_high = self.source_high.get_intensity()
        return torch.cat([i_low, i_high], dim=0)
    
    def set_intensity(self, intensity):
        """Set intensity for both exposures (expects [2] tensor)."""
        if intensity.numel() == 1:
            # If single value, apply to both
            self.source_low.set_intensity(intensity)
            self.source_high.set_intensity(intensity)
        else:
            # If two values, apply separately
            self.source_low.set_intensity(intensity[0:1])
            self.source_high.set_intensity(intensity[1:2])

    def forward(self, x):
        # Assume x has same shape as needed for both. 
        # Typically x is [n_rays, 1, 1] or similar dummy.
        q_low = self.source_low(x)   # [n_rays, 1, n_energies]
        q_high = self.source_high(x) # [n_rays, 1, n_energies]
        return torch.cat([q_low, q_high], dim=1) # [n_rays, 2, n_energies]

class MonoenergeticXraySource(UniformEmissionOperator):
    """
    Emission operator providing a single-energy (monoenergetic) spectrum.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, energies_keV, energy_val_keV, target_counts=1e6):
        energies_keV = torch.as_tensor(energies_keV, dtype=torch.float32)
        idx = torch.argmin(torch.abs(energies_keV - energy_val_keV))
        I0 = torch.zeros_like(energies_keV)
        I0[idx] = 1.0
        super().__init__(I0, target_counts=target_counts)

