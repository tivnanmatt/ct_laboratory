"""
Material database for common phantom materials.

Material composition definitions with elemental composition by mass fraction.

References:
[1] ICRU Report 44 (1989), Table 2 (NIST X-ray Mass Attenuation Coefficients)
[2] NIST Air composition: N (0.755268), O (0.231781), Ar (0.012827), C (0.000124)
[3] Soft Tissue (ICRU-44 Muscle / Skeletal): H (0.102), C (0.143), N (0.034), O (0.710), Na (0.001), P (0.002), S (0.003), Cl (0.001), K (0.004)
[4] Bone, Cortical (ICRU-44): H (0.034), C (0.155), N (0.042), O (0.435), Na (0.001), Mg (0.002), P (0.103), S (0.003), Ca (0.225)
[5] Cancellous Bone (ICRU-44-derived/Spongiosa): H (0.085), C (0.404), N (0.024), O (0.367), Mg (0.001), P (0.044), S (0.002), Ca (0.073)
"""

import numpy as np
import torch
from .mu_utils import get_mu


# Standard phantom materials with elemental composition and nominal properties
PHANTOM_MATERIALS = {
    "Air": {
        "comp": {"N": 0.755268, "O": 0.231781, "Ar": 0.012827, "C": 0.000124},
        "rho": 0.001205,
        "HU": -1000
    },
    "Water": {
        "comp": {"H": 0.111894, "O": 0.888106},
        "rho": 1.0,
        "HU": 0
    },
    "Soft Tissue": {  # Muscle, Skeletal (ICRU-44) - density will be rescaled to match HU
        "comp": {"H": 0.102, "C": 0.143, "N": 0.034, "O": 0.710, "Na": 0.001, "P": 0.002, "S": 0.003, "Cl": 0.001, "K": 0.004},
        "rho": 1.06,  # Initial density, will be rescaled
        "HU": 100
    },
    "Near Soft Tissue": {  # Same composition as Soft Tissue, different target HU
        "comp": {"H": 0.102, "C": 0.143, "N": 0.034, "O": 0.710, "Na": 0.001, "P": 0.002, "S": 0.003, "Cl": 0.001, "K": 0.004},
        "rho": 1.06,  # Initial density, will be rescaled
        "HU": 90
    },
    "Cancellous Bone": {  # Based on Spongiosa (ICRU-44 derivative)
        "comp": {"H": 0.085, "C": 0.404, "N": 0.024, "O": 0.367, "Mg": 0.001, "P": 0.044, "S": 0.002, "Ca": 0.073},
        "rho": 1.5,  # Initial density, will be rescaled
        "HU": 350
    },
    "Cortical Bone": {  # Based on Bone, Cortical (ICRU-44)
        "comp": {"H": 0.034, "C": 0.155, "N": 0.042, "O": 0.435, "Na": 0.001, "Mg": 0.002, "P": 0.103, "S": 0.003, "Ca": 0.225},
        "rho": 2.0,  # Initial density, will be rescaled
        "HU": 750
    },
    "Aluminium": {
        "comp": {"Al": 1.0},
        "rho": 2.7,  # Fixed density for Al - HU will be calculated from this
        "HU": None  # Will be calculated at reference energy
    },
}


class MaterialDatabase:
    """
    Database for looking up and computing material properties.
    
    Provides tools for:
    - Computing attenuation spectra for materials
    - Rescaling densities to match target HU values
    - Building interpolation tables for material properties
    """
    
    def __init__(self, materials=None, ref_energy_kev=90.0):
        """
        Initialize material database.
        
        Args:
            materials: Dictionary of materials (defaults to PHANTOM_MATERIALS)
            ref_energy_kev: Reference energy in keV for HU calculations
        """
        self.materials = materials if materials is not None else PHANTOM_MATERIALS
        self.ref_energy_kev = ref_energy_kev
        self.ref_energy_ev = ref_energy_kev * 1000.0
        
        # Water reference for HU calculations
        self.water_comp = {"H": 0.111894, "O": 0.888106}
        self.water_rho = 1.0
        
    def get_mu_ref_water(self):
        """Get reference attenuation coefficient of water at ref_energy."""
        return get_mu(self.water_comp, self.ref_energy_ev, self.water_rho)
    
    def compute_hu_from_mu(self, mu_material, mu_water=None):
        """
        Compute HU from attenuation coefficient.
        
        HU = 1000 * (mu_material - mu_water) / mu_water
        
        Args:
            mu_material: Attenuation coefficient of material at reference energy
            mu_water: Attenuation coefficient of water (computed if None)
        
        Returns:
            HU value
        """
        if mu_water is None:
            mu_water = self.get_mu_ref_water()
        return 1000.0 * (mu_material - mu_water) / mu_water
    
    def rescale_density_to_hu(self, comp, target_hu, initial_rho=1.0, max_iter=100, tol=0.01):
        """
        Iteratively rescale material density to match target HU.
        
        Args:
            comp: Elemental composition dict
            target_hu: Target HU value
            initial_rho: Initial density guess in g/cm^3
            max_iter: Maximum iterations for convergence
            tol: Tolerance for HU matching
        
        Returns:
            tuple: (rescaled_density, final_hu, converged)
        """
        mu_water_ref = self.get_mu_ref_water()
        rho = initial_rho
        
        for i in range(max_iter):
            mu_mat = get_mu(comp, self.ref_energy_ev, rho)
            hu_current = self.compute_hu_from_mu(mu_mat, mu_water_ref)
            
            if abs(hu_current - target_hu) < tol:
                return rho, hu_current, True
            
            # Newton-Raphson step: adjust density
            # d(HU)/d(rho) ≈ 1000 * d(mu)/d(rho) / mu_water
            # For linear scaling: mu ∝ rho, so d(mu)/d(rho) = mu/rho
            gradient = 1000.0 * (mu_mat / rho) / mu_water_ref
            delta_rho = (target_hu - hu_current) / (gradient + 1e-12)
            rho = max(rho + delta_rho, 0.001)  # Ensure positive density
        
        # Final check
        mu_mat = get_mu(comp, self.ref_energy_ev, rho)
        hu_final = self.compute_hu_from_mu(mu_mat, mu_water_ref)
        return rho, hu_final, abs(hu_final - target_hu) < tol * 10  # Relaxed tolerance
    
    def get_material_info(self, material_name):
        """Get material properties from database."""
        if material_name not in self.materials:
            raise ValueError(f"Material '{material_name}' not found in database")
        return self.materials[material_name]
    
    def compute_attenuation_spectrum(self, material_name, energies_ev, use_rescaled_density=True):
        """
        Compute energy-dependent attenuation spectrum for a material.
        
        Args:
            material_name: Name of material in database
            energies_ev: Array of energies in eV
            use_rescaled_density: If True and HU is specified, rescale density to match HU
        
        Returns:
            dict with 'mu_total', 'mu_pe', 'mu_cs', 'density', 'hu'
        """
        mat = self.get_material_info(material_name)
        comp = mat["comp"]
        rho = mat["rho"]
        target_hu = mat.get("HU")
        
        # Rescale density if HU is specified
        if use_rescaled_density and target_hu is not None:
            rho, _, _ = self.rescale_density_to_hu(comp, target_hu, initial_rho=rho)
        
        # Compute attenuation spectra
        if isinstance(energies_ev, (list, np.ndarray)):
            mu_total = np.array([get_mu(comp, e, rho) for e in energies_ev])
            mu_pe = np.array([get_mu(comp, e, rho, kind='photo') for e in energies_ev])
            mu_cs = np.array([get_mu(comp, e, rho, kind='incoh') + get_mu(comp, e, rho, kind='coh') for e in energies_ev])
        else:
            mu_total = get_mu(comp, energies_ev, rho)
            mu_pe = get_mu(comp, energies_ev, rho, kind='photo')
            mu_cs = get_mu(comp, energies_ev, rho, kind='incoh') + get_mu(comp, energies_ev, rho, kind='coh')
        
        # Compute actual HU at reference energy
        mu_ref = get_mu(comp, self.ref_energy_ev, rho) if isinstance(energies_ev, (list, np.ndarray)) else mu_total
        actual_hu = self.compute_hu_from_mu(mu_ref)
        
        return {
            'mu_total': mu_total,
            'mu_pe': mu_pe,
            'mu_cs': mu_cs,
            'density': rho,
            'hu': actual_hu
        }
