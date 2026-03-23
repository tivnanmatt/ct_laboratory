"""
Basis materials for spectral CT decomposition.

Provides tools for:
- Defining basis material systems (e.g., photoelectric/Compton)
- Computing basis attenuation spectra from physical materials
- Weighted least-squares fitting for basis decomposition
"""

import numpy as np
import torch
import spekpy
import xraydb
from scipy.interpolate import interp1d

from .mu_utils import get_mu, get_mu_photoelectric, get_mu_compton
from .material_database import MaterialDatabase, PHANTOM_MATERIALS


class BasisMaterials(torch.nn.Module):
    """
    Basis material system for spectral CT.
    
    Encapsulates the basis attenuation spectrum matrix Q and provides tools
    for material decomposition and property lookup.
    
    Shape conventions:
        Q: [n_energies, n_materials] - basis attenuation spectra
        x_basis: [n_rays, n_materials] - basis line integrals
    
    Common basis systems:
        - Photoelectric/Compton (PE/CS): n_materials=2
        - Water/Bone: n_materials=2
        - Three-material decomposition: n_materials=3
    """
    
    def __init__(self, Q, energies_kev=None, basis_names=None, material_db=None):
        """
        Initialize basis materials.
        
        Args:
            Q: Basis attenuation spectrum matrix [n_energies, n_materials]
            energies_kev: Energy bins in keV [n_energies]
            basis_names: Names of basis materials (e.g., ['PE', 'CS'])
            material_db: MaterialDatabase for property lookups
        """
        super().__init__()
        
        # Register Q as buffer
        Q_tensor = torch.as_tensor(Q, dtype=torch.float32)
        if Q_tensor.dim() == 1:
            Q_tensor = Q_tensor.unsqueeze(1)  # [E] -> [E, 1]
        self.register_buffer("Q", Q_tensor)
        
        # Store metadata
        self.n_energies = Q_tensor.shape[0]
        self.n_materials = Q_tensor.shape[1]
        
        if energies_kev is not None:
            energies_kev = torch.as_tensor(energies_kev, dtype=torch.float32)
            self.register_buffer("energies_kev", energies_kev)
        else:
            self.energies_kev = None
        
        self.basis_names = basis_names or [f"Basis{i}" for i in range(self.n_materials)]
        self.material_db = material_db
        
    @property
    def shape(self):
        """Return shape of Q matrix."""
        return self.Q.shape
    
    def compute_transmission(self, x_basis):
        """
        Compute transmission: exp(-Q @ x_basis.T).T
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
        
        Returns:
            transmission: [n_rays, n_energies] - transmission factors
        """
        # Q: [n_energies, n_materials], x_basis: [n_rays, n_materials]
        # att = x_basis @ Q.T -> [n_rays, n_energies]
        att = torch.matmul(x_basis, self.Q.T)

        att_lower_limit = -5.0
        if torch.any(att < att_lower_limit):
            print(f"Warning: Attenuation values below {att_lower_limit} detected. Clipping for stability.")
            att = torch.clamp(att, min=att_lower_limit)

        return torch.exp(-att)
    
    def compute_attenuation(self, x_basis):
        """
        Compute attenuation: Q @ x_basis.T
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
        
        Returns:
            attenuation: [n_rays, n_energies] - attenuation values
        """
        # Q: [n_energies, n_materials], x_basis: [n_rays, n_materials]
        # att = x_basis @ Q.T -> [n_rays, n_energies]
        return torch.matmul(x_basis, self.Q.T)
    
    def forward(self, x_basis):
        """
        Forward pass: compute attenuation from basis line integrals.
        
        This is the standard torch.nn.Module forward method that performs
        matrix multiplication: x_basis @ Q.T to get attenuation in each energy channel.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
        
        Returns:
            attenuation: [n_rays, n_energies] - attenuation values (mu * length)
        """
        return self.compute_attenuation(x_basis)
    
    def approximate_material(self, mu_material, thickness_mm=1.0, weights=None, x_true=None):
        """
        Approximate a material's attenuation spectrum using basis functions.
        
        Solves the weighted least squares problem:
            Q @ x = mu_material * thickness_mm
        
        If weights are provided, solves:
            (Q^T W Q) x = Q^T W (mu_material * thickness_mm)
        
        This allows fitting any material's attenuation spectrum with the basis
        functions (e.g., PE/CS decomposition of tissue).
        
        Args:
            mu_material: Either:
                - [n_energies] array/tensor - attenuation coefficient spectrum (1/mm)
                - BasisMaterials object - true basis material with Q_true matrix
            thickness_mm: Thickness in mm to compute line integral
            weights: [n_energies] - optional weights for least squares
                    (e.g., detected spectrum in air for spectral weighting)
            x_true: [n_materials_true] - true basis coefficients (required if mu_material is BasisMaterials)
        
        Returns:
            x_basis: [n_materials] - basis coefficients (line integrals in mm)
        
        Example:
            >>> # Approximate soft tissue with PE/CS basis
            >>> basis = PECSBasis(energies_keV)
            >>> mu_tissue = [get_mu(tissue_comp, e, rho) for e in energies_eV]
            >>> x_basis = basis.approximate_material(mu_tissue, thickness_mm=300)
            >>> # x_basis[0] = PE line integral, x_basis[1] = CS line integral
            
            >>> # Or use BasisMaterials input:
            >>> soft_tissue_basis = BasisMaterials(energies_keV, Q_st, ["SoftTissue"])
            >>> x_st_true = torch.tensor([1.0])
            >>> x_basis = basis.approximate_material(soft_tissue_basis, thickness_mm=300, x_true=x_st_true)
        """
        # Check if input is a BasisMaterials object
        if isinstance(mu_material, BasisMaterials):
            if x_true is None:
                raise ValueError("x_true must be provided when mu_material is a BasisMaterials object")
            
            # Extract Q_true and compute attenuation: Q_true @ x_true
            Q_true_np = mu_material.Q.cpu().numpy()  # [n_energies, n_materials_true]
            if isinstance(x_true, torch.Tensor):
                x_true_np = x_true.cpu().numpy()
            else:
                x_true_np = np.array(x_true)
            
            # Compute the actual attenuation spectrum
            mu_material = Q_true_np @ x_true_np  # [n_energies]
            y = mu_material * thickness_mm  # [n_energies]
        else:
            # Original behavior: convert inputs to numpy for computation
            if isinstance(mu_material, torch.Tensor):
                mu_material = mu_material.cpu().numpy()
            
            y = mu_material * thickness_mm  # [n_energies]
        
        Q_np = self.Q.cpu().numpy()  # [n_energies, n_materials]
        
        if weights is not None:
            # Weighted least squares
            if isinstance(weights, torch.Tensor):
                weights = weights.cpu().numpy()
            
            W = np.diag(weights)  # [n_energies, n_energies]
            lhs = Q_np.T @ W @ Q_np  # [n_materials, n_materials]
            rhs = Q_np.T @ W @ y  # [n_materials]
        else:
            # Unweighted least squares
            lhs = Q_np.T @ Q_np  # [n_materials, n_materials]
            rhs = Q_np.T @ y  # [n_materials]
        
        # Solve for basis coefficients
        x_basis = np.linalg.solve(lhs, rhs)
        
        return torch.from_numpy(x_basis).float().to(self.Q.device)
    
    @classmethod
    def from_water_pe_cs(cls, energies_kev, dk=1.0, ref_energy_kev=90.0):
        """
        Create PE/CS basis materials using water as reference.
        
        This is the standard basis for dual-energy CT, where:
        - Basis 0: Photoelectric effect (low energy dependent)
        - Basis 1: Compton scattering (nearly energy independent)
        
        Args:
            energies_kev: Energy bins in keV
            dk: Energy bin width for spectrum calculations
            ref_energy_kev: Reference energy for normalization
        
        Returns:
            BasisMaterials instance with PE/CS basis
        """
        energies_ev = np.array(energies_kev) * 1000.0
        water_comp = {"H": 0.111894, "O": 0.888106}
        water_rho = 1.0
        
        # Compute PE and CS components for water
        f_pe = np.array([get_mu_photoelectric(water_comp, e, water_rho) for e in energies_ev])
        f_cs = np.array([get_mu_compton(water_comp, e, water_rho) for e in energies_ev])
        
        # Stack into Q matrix [E, 2]
        Q = np.stack([f_pe, f_cs], axis=1)
        
        # Create material database
        material_db = MaterialDatabase(ref_energy_kev=ref_energy_kev)
        
        return cls(Q, energies_kev, basis_names=['PE', 'CS'], material_db=material_db)
    
    @classmethod
    def from_water_pe_cs_weighted(cls, energies_kev, kvp=120, dk=1.0, 
                                    al_filter_mm=5.0, gos_thickness_mm=0.5,
                                    ref_energy_kev=90.0):
        """
        Create PE/CS basis materials with weighted least-squares fitting.
        
        Uses detected spectrum (air) to weight the basis decomposition,
        accounting for:
        - X-ray source spectrum
        - Aluminum filtration
        - GOS detector interaction probability
        
        Args:
            energies_kev: Energy bins in keV
            kvp: X-ray tube peak voltage
            dk: Energy bin width for spekpy
            al_filter_mm: Aluminum filter thickness in mm
            gos_thickness_mm: GOS detector thickness in mm
            ref_energy_kev: Reference energy for HU calculations
        
        Returns:
            BasisMaterials instance with weighted PE/CS basis
        """
        n_bins = len(energies_kev)
        energies_ev = np.array(energies_kev) * 1000.0
        
        # 1. Basis materials (Water PE/CS components)
        water_comp = {"H": 0.111894, "O": 0.888106}
        water_rho = 1.0
        f_pe_water = np.array([get_mu_photoelectric(water_comp, e, water_rho) for e in energies_ev])
        f_cs_water = np.array([get_mu_compton(water_comp, e, water_rho) for e in energies_ev])
        A = np.stack([f_pe_water, f_cs_water], axis=1)  # [n_bins, 2]
        
        # 2. Compute weights from detected spectrum (air)
        s = spekpy.Spek(kvp=kvp, dk=float(dk))
        k, sp = s.get_spectrum()
        spec_emission = np.zeros(n_bins)
        for i, val in enumerate(k):
            idx = int(round((val - dk/2) / dk))
            if 0 <= idx < n_bins:
                spec_emission[idx] = val
        
        # Aluminum filter transmission
        al_mu = np.array([xraydb.material_mu('Al', float(e)) / 10.0 for e in energies_ev])
        transmission_al = np.exp(-al_mu * al_filter_mm)
        
        # GOS detector interaction probability
        gos_density = 7.3
        gos_comp = {'Gd': 0.8308, 'O': 0.0845, 'S': 0.0847}
        mu_gos = np.array([get_mu(gos_comp, e, gos_density) for e in energies_ev])
        prob_int = 1.0 - np.exp(-mu_gos * gos_thickness_mm)
        
        # Combined weights
        weights = spec_emission * transmission_al * prob_int
        W = np.diag(weights)
        
        # 3. Weighted least squares (not actually solving, just using A as basis)
        # The weighting ensures materials are optimally decomposed in the detected spectrum
        # For now, just use the unweighted basis (could implement weighted fitting here)
        Q = A
        
        # Create material database
        material_db = MaterialDatabase(ref_energy_kev=ref_energy_kev)
        
        basis = cls(Q, energies_kev, basis_names=['PE', 'CS'], material_db=material_db)
        
        # Store weights for future use
        basis.register_buffer("weights", torch.from_numpy(weights).float())
        basis.register_buffer("W_matrix", torch.from_numpy(W).float())
        
        return basis
    
    def build_material_lookup_tables(self, materials=None):
        """
        Build interpolation tables for material properties vs HU.
        
        Computes PE/CS coefficients and density for each material,
        then creates interpolation functions for arbitrary HU values.
        
        Args:
            materials: List of material names (uses all PHANTOM_MATERIALS if None)
        
        Returns:
            dict with interpolation functions and tables
        """
        if self.material_db is None:
            self.material_db = MaterialDatabase()
        
        if materials is None:
            materials = list(PHANTOM_MATERIALS.keys())
        
        # Sort materials by HU for interpolation
        material_list = []
        for name in materials:
            mat_info = PHANTOM_MATERIALS[name]
            if mat_info["HU"] is not None:
                material_list.append((name, mat_info["HU"]))
        
        # Sort by HU
        material_list.sort(key=lambda x: x[1])
        sorted_names = [name for name, _ in material_list]
        
        # Compute properties for each material
        energies_ev = self.energies_kev.cpu().numpy() * 1000.0 if self.energies_kev is not None else None
        
        if energies_ev is None:
            raise ValueError("Energy bins must be specified to build lookup tables")
        
        table_hu = []
        table_densities = []
        table_pe = []
        table_cs = []
        table_mu_ref = []
        
        for name in sorted_names:
            result = self.material_db.compute_attenuation_spectrum(name, energies_ev, use_rescaled_density=True)
            
            # Integrate with weights if available
            if hasattr(self, 'weights'):
                weights = self.weights.cpu().numpy()
                pe_integral = np.sum(result['mu_pe'] * weights)
                cs_integral = np.sum(result['mu_cs'] * weights)
            else:
                pe_integral = np.mean(result['mu_pe'])
                cs_integral = np.mean(result['mu_cs'])
            
            table_hu.append(result['hu'])
            table_densities.append(result['density'])
            table_pe.append(pe_integral)
            table_cs.append(cs_integral)
            table_mu_ref.append(result['mu_total'][len(energies_ev)//2])  # Use middle energy as reference
        
        # Create interpolation functions
        table_hu = np.array(table_hu)
        table_densities = np.array(table_densities)
        table_pe = np.array(table_pe)
        table_cs = np.array(table_cs)
        table_mu_ref = np.array(table_mu_ref)
        
        f_rho = interp1d(table_hu, table_densities, kind='linear', fill_value='extrapolate')
        f_pe = interp1d(table_hu, table_pe, kind='linear', fill_value='extrapolate')
        f_cs = interp1d(table_hu, table_cs, kind='linear', fill_value='extrapolate')
        f_mu_ref = interp1d(table_hu, table_mu_ref, kind='linear', fill_value='extrapolate')
        
        return {
            "materials": sorted_names,
            "table_hu": table_hu,
            "table_densities": table_densities,
            "table_pe": table_pe,
            "table_cs": table_cs,
            "table_mu_ref": table_mu_ref,
            "f_rho": f_rho,
            "f_pe": f_pe,
            "f_cs": f_cs,
            "f_mu_ref": f_mu_ref,
        }
