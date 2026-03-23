"""
Basis attenuation operator.

This operator converts basis line integrals into transmission factors using
a basis material system. It's used for modeling object attenuation in spectral CT.

The core operation is: transmission = exp(-Q @ x_basis^T)

where:
- Q: [n_energies, n_basis] - basis attenuation spectra
- x_basis: [n_rays, n_basis] - basis line integrals
- transmission: [n_rays, n_energies] - transmission factors
"""

import torch


class BasisAttenuator(torch.nn.Module):
    """
    Attenuation operator using basis materials.
    
    Computes transmission from basis line integrals:
        transmission = exp(-Q @ x_basis^T)
    
    This is the fundamental operation for spectral CT forward modeling,
    converting line integrals in basis space (e.g., PE/CS) to energy-dependent
    transmission.
    
    Shape conventions:
        Q: [n_energies, n_materials] - basis attenuation spectra
        x_basis: [n_rays, n_materials] - basis line integrals
        transmission: [n_rays, n_energies] - transmission factors
    """
    
    def __init__(self, basis_materials):
        """
        Initialize basis attenuator.
        
        Args:
            basis_materials: BasisMaterials object containing Q matrix
        """
        super().__init__()
        self.basis_materials = basis_materials
    
    @property
    def Q(self):
        """Access to basis attenuation spectrum matrix."""
        return self.basis_materials.Q
    
    @property
    def n_materials(self):
        """Number of basis materials."""
        return self.basis_materials.n_materials
    
    @property
    def n_energies(self):
        """Number of energy bins."""
        return self.basis_materials.n_energies
    
    def compute_attenuation(self, x_basis):
        """
        Compute line attenuation: Q @ x_basis^T
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
        
        Returns:
            attenuation: [n_rays, n_energies] - attenuation values
        """
        return self.basis_materials.compute_attenuation(x_basis)
    
    def compute_transmission(self, x_basis):
        """
        Compute transmission: exp(-Q @ x_basis^T)
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
        
        Returns:
            transmission: [n_rays, n_energies] - transmission factors
        """
        return self.basis_materials.compute_transmission(x_basis)
    
    def forward(self, x_basis):
        """
        Apply basis attenuation.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
        
        Returns:
            transmission: [n_rays, n_energies] - transmission factors
        """
        return self.compute_transmission(x_basis)


class ObjectAttenuator(torch.nn.Module):
    """
    Object attenuation operator for use in XraySystem.
    
    This operator takes a spectrum and basis integrals, applies the object
    attenuation, and returns the attenuated spectrum.
    
    Shape conventions:
        Input q0: [n_rays, n_exposures, n_energies] - input spectrum
        x_basis: [n_rays, n_basis] - basis line integrals
        Output q: [n_rays, n_exposures, n_energies] - attenuated spectrum
    """
    
    def __init__(self, basis_materials):
        """
        Initialize object attenuator.
        
        Args:
            basis_materials: BasisMaterials object containing Q matrix
        """
        super().__init__()
        self.basis_attenuator = BasisAttenuator(basis_materials)
    
    def forward(self, q0, x_basis):
        """
        Apply object attenuation to spectrum.
        
        Args:
            q0: Input spectrum [n_rays, n_exposures, n_energies]
            x_basis: Basis line integrals [n_rays, n_materials]
        
        Returns:
            q: Attenuated spectrum [n_rays, n_exposures, n_energies]
        """
        # Compute transmission: [n_rays, n_energies]
        transmission = self.basis_attenuator.compute_transmission(x_basis)
        
        # Broadcast to [n_rays, 1, n_energies] for multiplication with exposures
        transmission = transmission.unsqueeze(1)
        
        # Apply attenuation
        return q0 * transmission
