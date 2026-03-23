"""
Fixed-length attenuation operators.

These operators apply a pre-computed transmission factor, typically used for:
- Beam filters (e.g., aluminum, copper)
- Bow-tie filters
- Any fixed attenuation in the beam path

The transmission is computed once during initialization and applied uniformly
(or ray-dependently if specified).
"""

import torch
import xraydb


class FixedAttenuator(torch.nn.Module):
    """
    Base class for fixed-length attenuation.
    
    Applies a pre-computed transmission factor to the input spectrum.
    
    Shape conventions:
        Input q0: [n_rays, n_exposures, n_energies]
        Output q: [n_rays, n_exposures, n_energies]
        transmission: [1, 1, n_energies] (broadcasts) or [n_rays, 1, n_energies] (ray-dependent)
    """
    
    def __init__(self, transmission):
        """
        Initialize with a transmission factor.
        
        Args:
            transmission: Transmission factor [1, 1, E] or [n_rays, 1, E]
        """
        super().__init__()
        transmission_tensor = torch.as_tensor(transmission, dtype=torch.float32)
        
        # Ensure proper broadcasting shape
        if transmission_tensor.dim() == 1:
            transmission_tensor = transmission_tensor.view(1, 1, -1)
        elif transmission_tensor.dim() == 2:
            transmission_tensor = transmission_tensor.unsqueeze(1)
        
        self.register_buffer("transmission", transmission_tensor)
    
    def forward(self, q0):
        """
        Apply attenuation to input spectrum.
        
        Args:
            q0: Input spectrum [n_rays, n_exposures, n_energies]
        
        Returns:
            q: Attenuated spectrum [n_rays, n_exposures, n_energies]
        """
        return q0 * self.transmission
    
    def sample(self, q0):
        """
        Apply attenuation and sample with Poisson noise.
        
        Args:
            q0: Input spectrum [n_rays, n_exposures, n_energies]
        
        Returns:
            q_sample: Poisson-sampled attenuated spectrum [n_rays, n_exposures, n_energies]
        """
        q_mean = self.forward(q0)
        return torch.poisson(torch.clamp(q_mean, min=0.0))


class UniformAluminumFilter(FixedAttenuator):
    """
    Aluminum filter with uniform thickness.
    
    Computes transmission through aluminum using xraydb, then applies it uniformly
    to all rays and exposures.
    
    This is a common beam filter in X-ray systems.
    """
    
    def __init__(self, energies_ev, thickness_mm):
        """
        Initialize aluminum filter.
        
        Args:
            energies_ev: Energy bins in eV [n_energies]
            thickness_mm: Aluminum thickness in mm
        """
        # Compute attenuation coefficient for Al
        # xraydb returns in cm^-1, divide by 10 for mm^-1
        energies_ev = torch.as_tensor(energies_ev, dtype=torch.float32)
        al_mu = torch.tensor(
            [xraydb.material_mu('Al', e.item()) / 10.0 for e in energies_ev],
            dtype=torch.float32
        )
        
        # Compute transmission: exp(-mu * thickness)
        transmission = torch.exp(-al_mu * thickness_mm)
        
        # Store thickness as parameter for reference
        super().__init__(transmission)
        self.thickness_mm = thickness_mm
        self.register_buffer("energies_ev", energies_ev)
        self.register_buffer("al_mu", al_mu)


class MaterialFilter(FixedAttenuator):
    """
    General material filter with fixed thickness.
    
    Can be used for any material defined in xraydb or with custom composition.
    """
    
    def __init__(self, energies_ev, material_comp, thickness_mm, density=None):
        """
        Initialize material filter.
        
        Args:
            energies_ev: Energy bins in eV [n_energies]
            material_comp: Material composition (str or dict)
            thickness_mm: Material thickness in mm
            density: Material density in g/cm^3 (required if comp is dict)
        """
        from ..mu_utils import get_mu
        
        energies_ev = torch.as_tensor(energies_ev, dtype=torch.float32)
        
        # Compute attenuation coefficients
        if isinstance(material_comp, str):
            mu = torch.tensor(
                [xraydb.material_mu(material_comp, e.item()) / 10.0 for e in energies_ev],
                dtype=torch.float32
            )
        else:
            if density is None:
                raise ValueError("Density must be specified for elemental compositions")
            mu = torch.tensor(
                [get_mu(material_comp, e.item(), density) for e in energies_ev],
                dtype=torch.float32
            )
        
        # Compute transmission
        transmission = torch.exp(-mu * thickness_mm)
        
        super().__init__(transmission)
        self.material_comp = material_comp
        self.thickness_mm = thickness_mm
        self.density = density
        self.register_buffer("energies_ev", energies_ev)
        self.register_buffer("mu", mu)


class RayDependentFilter(FixedAttenuator):
    """
    Ray-dependent filter (e.g., bow-tie filter).
    
    Each ray can have a different transmission factor, typically used for
    modeling filters with varying thickness across the field of view.
    """
    
    def __init__(self, transmission_per_ray):
        """
        Initialize ray-dependent filter.
        
        Args:
            transmission_per_ray: Transmission for each ray [n_rays, n_energies]
        """
        transmission_per_ray = torch.as_tensor(transmission_per_ray, dtype=torch.float32)
        
        # Add exposure dimension: [n_rays, n_energies] -> [n_rays, 1, n_energies]
        if transmission_per_ray.dim() == 2:
            transmission_per_ray = transmission_per_ray.unsqueeze(1)
        
        super().__init__(transmission_per_ray)
