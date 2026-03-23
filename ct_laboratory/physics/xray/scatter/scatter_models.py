import torch

class ZeroScatterModel(torch.nn.Module):
    """
    Zero scatter model that returns no scatter.
    
    Output: [n_rays, n_exposures, n_energies]
    """
    def __init__(self):
        super().__init__()
        self.basis_materials = None
        self.emission_op = None
        self.filter_op = None
        self._is_calibrated = False
    
    def calibrate_scatter(self, basis_materials, emission_op, filter_op):
        """
        Calibrate scatter model with system components.
        
        Args:
            basis_materials: BasisMaterials object
            emission_op: Emission operator
            filter_op: Filter operator
        """
        self.basis_materials = basis_materials
        self.emission_op = emission_op
        self.filter_op = filter_op
        self._is_calibrated = True
        return self
    
    def forward(self, x_basis, q_attenuated=None):
        """
        Compute scattered photons.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            q_attenuated: [n_rays, n_exposures, n_energies] - optional attenuated spectrum (optimization hook)
            
        Returns:
            scatter: [n_rays, n_exposures, n_energies] - scattered photons (all zeros)
        """
        if q_attenuated is not None:
            return torch.zeros_like(q_attenuated)
        
        # If no hook provided and calibrated, create zero scatter with correct shape
        if self._is_calibrated and self.basis_materials is not None:
            n_rays = x_basis.shape[0]
            n_energies = self.basis_materials.n_energies
            device = x_basis.device
            # Assume single exposure for default shape
            return torch.zeros(n_rays, 1, n_energies, device=device)
        
        # Fallback: return zeros matching x_basis shape
        n_rays = x_basis.shape[0]
        return torch.zeros(n_rays, 1, 1, device=x_basis.device)

class ConstantScatterModel(torch.nn.Module):
    """
    Constant scatter model with spectral shape matching attenuated primary beam.
    
    The scatter has a constant magnitude (target_scatter_photons) but its
    spectral distribution matches the attenuated primary beam q_attenuated.
    
    Output: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, target_scatter_photons=1e2):
        super().__init__()
        self.target_scatter_photons = target_scatter_photons
        self.basis_materials = None
        self.emission_op = None
        self.filter_op = None
        self._is_calibrated = False
    
    def calibrate_scatter(self, basis_materials, emission_op, filter_op):
        """
        Calibrate scatter model with system components.
        
        Args:
            basis_materials: BasisMaterials object
            emission_op: Emission operator
            filter_op: Filter operator
        """
        self.basis_materials = basis_materials
        self.emission_op = emission_op
        self.filter_op = filter_op
        self._is_calibrated = True
        return self
    
    def forward(self, x_basis, q_attenuated=None):
        """
        Compute scattered photons with spectral shape matching q_attenuated.
        
        Args:
            x_basis: [n_rays, n_materials] - basis line integrals
            q_attenuated: [n_rays, n_exposures, n_energies] - optional attenuated spectrum (optimization hook)
                         If provided, uses this directly. Otherwise computes it internally.
            
        Returns:
            scatter: [n_rays, n_exposures, n_energies] - scattered photons
        """
        # If q_attenuated provided as optimization hook, use it directly
        if q_attenuated is None:
            # Compute q_attenuated from x_basis
            if not self._is_calibrated:
                raise RuntimeError("ConstantScatterModel must be calibrated before use. Call calibrate_scatter().")
            
            # Reproduce the forward pass to get q_attenuated
            n_rays = x_basis.shape[0]
            n_energies = self.basis_materials.n_energies
            device = x_basis.device
            
            # Emission
            dummy = torch.ones(n_rays, 1, n_energies, device=device)
            q_emission = self.emission_op(dummy)
            
            # Filter
            q_filtered = self.filter_op(q_emission)
            
            # Object attenuation
            transmission = self.basis_materials.compute_transmission(x_basis)
            transmission = transmission.unsqueeze(1)  # [n_rays, 1, n_energies]
            q_attenuated = q_filtered * transmission
        
        # Scatter has same spectral shape as attenuated beam, but with constant total magnitude
        scatter_shape = q_attenuated / (q_attenuated.sum(dim=-1, keepdim=True) + 1e-12)
        scatter = scatter_shape * self.target_scatter_photons
        
        return scatter
