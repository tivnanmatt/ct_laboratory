"""
Utility functions for computing attenuation coefficients using xraydb.
"""

import numpy as np
import xraydb


def get_mu(comp, energy_ev, density=1.0, kind='total'):
    """
    Get mass attenuation coefficient in mm^-1.
    
    Args:
        comp: Material composition. Can be:
            - str: Material name (e.g., 'Water', 'Al')
            - dict: Elemental composition by mass fraction (e.g., {'H': 0.111894, 'O': 0.888106})
        energy_ev: Energy in eV (scalar or array-like)
        density: Material density in g/cm^3
        kind: Type of attenuation coefficient:
            - 'total': Total attenuation
            - 'photo': Photoelectric absorption
            - 'incoh': Incoherent (Compton) scattering
            - 'coh': Coherent (Rayleigh) scattering
    
    Returns:
        Attenuation coefficient in mm^-1 (linear attenuation, not mass attenuation)
    """
    # Fix for xraydb compatibility with numpy types
    if isinstance(energy_ev, (np.ndarray, list)):
        energy_ev = np.asanyarray(energy_ev).astype(float)
    elif hasattr(energy_ev, '__iter__'): # other iterables
        energy_ev = [float(e) for e in energy_ev]
    else:
        energy_ev = float(energy_ev)

    if isinstance(comp, str):
        # For named materials, use xraydb.material_mu
        # Returns in cm^-1, so divide by 10 to get mm^-1
        return xraydb.material_mu(comp, energy_ev, density=density, kind=kind) / 10.0
    else:
        # For elemental compositions, compute weighted sum
        mu_rho = 0.0  # Mass attenuation coefficient in cm^2/g
        for symbol, fraction in comp.items():
            mu_rho += fraction * xraydb.mu_elam(symbol, energy_ev, kind=kind)
        # Convert to linear attenuation in mm^-1: mu_rho * rho / 10
        return (mu_rho * density) / 10.0


def get_mu_photoelectric(comp, energy_ev, density=1.0):
    """
    Get photoelectric mass attenuation coefficient in mm^-1.
    
    Args:
        comp: Material composition (str or dict)
        energy_ev: Energy in eV (scalar or array-like)
        density: Material density in g/cm^3
    
    Returns:
        Photoelectric attenuation coefficient in mm^-1
    """
    return get_mu(comp, energy_ev, density, kind='photo')


def get_mu_compton(comp, energy_ev, density=1.0):
    """
    Get Compton scattering mass attenuation coefficient in mm^-1.
    
    Combines incoherent (Compton) and coherent (Rayleigh) scattering.
    
    Args:
        comp: Material composition (str or dict)
        energy_ev: Energy in eV (scalar or array-like)
        density: Material density in g/cm^3
    
    Returns:
        Compton scattering attenuation coefficient in mm^-1
    """
    mu_incoh = get_mu(comp, energy_ev, density, kind='incoh')
    mu_coh = get_mu(comp, energy_ev, density, kind='coh')
    return mu_incoh + mu_coh


def get_mu_material(material_name, energies_ev):
    """
    Get attenuation coefficient spectrum for a named material from the database.
    
    Convenience function that looks up material properties from PHANTOM_MATERIALS
    and computes attenuation coefficients for all energies.
    
    Args:
        material_name: Name of material (e.g., 'Soft Tissue', 'Water', 'Cortical Bone')
        energies_ev: Energy array in eV (scalar or array-like)
    
    Returns:
        Attenuation coefficient spectrum in mm^-1 (numpy array if energies_ev is array)
    
    Example:
        >>> energies_eV = np.linspace(20000, 120000, 100)
        >>> mu_tissue = get_mu_material('Soft Tissue', energies_eV)
    """
    from .material_database import PHANTOM_MATERIALS
    
    if material_name not in PHANTOM_MATERIALS:
        available = ', '.join(PHANTOM_MATERIALS.keys())
        raise ValueError(f"Material '{material_name}' not found. Available materials: {available}")
    
    props = PHANTOM_MATERIALS[material_name]
    
    # Handle both scalar and array energies
    if np.isscalar(energies_ev):
        return get_mu(props["comp"], energies_ev, props["rho"])
    else:
        return np.array([get_mu(props["comp"], e, props["rho"]) for e in energies_ev])
