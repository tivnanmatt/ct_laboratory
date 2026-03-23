"""
Standard basis material definitions for spectral CT decomposition.

This module provides pre-defined basis materials for common decomposition tasks:
- PE/CS (Photoelectric/Compton) basis for material decomposition
- Other standard bases can be added here
"""

import numpy as np
from .basis_materials import BasisMaterials
from .mu_utils import get_mu_photoelectric, get_mu_compton, get_mu_material


def PECSBasis(energies_keV, reference_material=None, reference_density=1.0):
    """
    Create a PE/CS (Photoelectric/Compton Scattering) basis for material decomposition.
    
    This is the standard two-material basis used in spectral CT, representing:
    - Photoelectric effect (PE): dominant at low energies
    - Compton scattering (CS): dominant at high energies (includes coherent/Rayleigh)
    
    Args:
        energies_keV: Energy bins in keV [n_energies]
        reference_material: Material composition dict (default: water)
        reference_density: Density of reference material in g/cm³ (default: 1.0 for water)
        
    Returns:
        BasisMaterials object with PE and CS basis spectra
        
    Example:
        >>> energies_keV = np.arange(0.5, 150, 1.0)
        >>> basis = PECSBasis(energies_keV)
        >>> # Use in XraySystem
        >>> xray_system = XraySystem(energies_keV, basis, ...)
    """
    if reference_material is None:
        # Default to water composition (H2O)
        reference_material = {"H": 0.111894, "O": 0.888106}
    
    energies_keV = np.asarray(energies_keV, dtype=np.float64)
    energies_eV = energies_keV * 1000.0
    
    # Compute basis functions for reference material (typically water)
    f_pe = np.array([get_mu_photoelectric(reference_material, e, reference_density) 
                     for e in energies_eV])
    f_cs = np.array([get_mu_compton(reference_material, e, reference_density) 
                     for e in energies_eV])
    
    # Stack into Q matrix [E, 2] (energies x basis)
    Q_matrix = np.stack([f_pe, f_cs], axis=1)
    
    return BasisMaterials(
        Q=Q_matrix,
        energies_kev=energies_keV,
        basis_names=['PE', 'CS']
    )


def WaterIodineBasis(energies_keV):
    """
    Create a Water/Iodine basis for contrast-enhanced CT.
    
    This basis separates water-equivalent soft tissue from iodine contrast agent.
    Can be implemented in future versions.
    
    Args:
        energies_keV: Energy bins in keV [n_energies]
        
    Returns:
        BasisMaterials object with Water and Iodine basis spectra
    """
    raise NotImplementedError("Water/Iodine basis not yet implemented")


def WaterCalciumBasis(energies_keV):
    """
    Create a Water/Calcium basis for bone imaging.
    
    This basis separates water-equivalent soft tissue from calcium (bone).
    Can be implemented in future versions.
    
    Args:
        energies_keV: Energy bins in keV [n_energies]
        
    Returns:
        BasisMaterials object with Water and Calcium basis spectra
    """
    raise NotImplementedError("Water/Calcium basis not yet implemented")


class WaterBasis(BasisMaterials):
    """
    Single-material basis for water.
    
    This creates a BasisMaterials object with one basis function representing
    the attenuation spectrum of water. Useful for direct water calculations
    or as a reference for approximating with PE/CS bases.
    
    Example:
        >>> energies_keV = np.arange(0.5, 150, 1.0)
        >>> water_basis = WaterBasis(energies_keV)
        >>> # Water basis has shape [n_energies, 1]
    """
    def __init__(self, energies_keV, density=1.0):
        """
        Initialize water basis.
        
        Args:
            energies_keV: Energy bins in keV [n_energies]
            density: Water density in g/cm³ (default: 1.0)
        """
        energies_keV = np.asarray(energies_keV, dtype=np.float64)
        energies_eV = energies_keV * 1000.0
        
        # Water composition
        water_comp = {"H": 0.111894, "O": 0.888106}
        
        # Get material attenuation spectrum
        from .mu_utils import get_mu
        mu_water = np.array([get_mu(water_comp, e, density) for e in energies_eV])
        
        # Create Q matrix [n_energies, 1]
        Q_matrix = mu_water.reshape(-1, 1)
        
        super().__init__(
            Q=Q_matrix,
            energies_kev=energies_keV,
            basis_names=["Water"]
        )


class SoftTissueBasis(BasisMaterials):
    """
    Single-material basis for a specific tissue type.
    
    This creates a BasisMaterials object with one basis function representing
    the attenuation spectrum of a specific material (e.g., Soft Tissue).
    Useful for approximating materials with PE/CS or other bases.
    
    Example:
        >>> energies_keV = np.arange(0.5, 150, 1.0)
        >>> st_basis = SoftTissueBasis(energies_keV)
        >>> x_true = torch.tensor([1.0])  # Identity coefficient
        >>> # Approximate with PE/CS basis
        >>> pecs_basis = PECSBasis(energies_keV)
        >>> x_approx = pecs_basis.approximate_material(st_basis, thickness_mm=300.0, x_true=x_true)
    """
    def __init__(self, energies_keV, material_name="Soft Tissue"):
        """
        Initialize soft tissue basis.
        
        Args:
            energies_keV: Energy bins in keV [n_energies]
            material_name: Name of material in PHANTOM_MATERIALS database
        """
        energies_keV = np.asarray(energies_keV, dtype=np.float64)
        energies_eV = energies_keV * 1000.0
        
        # Get material attenuation spectrum
        mu_material = get_mu_material(material_name, energies_eV)
        
        # Create Q matrix [n_energies, 1]
        Q_matrix = mu_material.reshape(-1, 1)
        
        super().__init__(
            Q=Q_matrix,
            energies_kev=energies_keV,
            basis_names=[material_name.replace(" ", "")]
        )
