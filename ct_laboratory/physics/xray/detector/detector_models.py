import torch
from ct_laboratory.physics.xray.attenuation.mu_utils import get_mu

class UniformGOSInteraction(torch.nn.Module):
    """
    A module modeling the GOS detector interaction.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self, energies_eV, thickness_mm):
        super().__init__()
        gos_density = 7.3
        gos_comp = {'Gd': 0.8308, 'O': 0.0845, 'S': 0.0847}
        gos_mu = torch.tensor([get_mu(gos_comp, e.item(), gos_density) for e in energies_eV], dtype=torch.float32)
        prob_int = (1.0 - torch.exp(-gos_mu * thickness_mm)).view(1, 1, -1)
        self.register_buffer("prob_int", prob_int)
    
    def forward(self, q_incident):
        """Apply interaction probability to incident spectrum."""
        return q_incident * self.prob_int
    
    def sample(self, q_incident):
        """
        Apply interaction probability and sample with Poisson noise.
        
        Args:
            q_incident: Incident spectrum [n_rays, n_exposures, n_energies]
            
        Returns:
            q_detected: Poisson-sampled detected spectrum [n_rays, n_exposures, n_energies]
        """
        q_mean = self.forward(q_incident)
        return torch.poisson(torch.clamp(q_mean, min=0.0))

class EnergyIntegratingCTDetector(torch.nn.Module):
    """
    A CT detector that integrates spectral intensity into a single signal per exposure.
    
    Output Shape: [n_rays, n_channels]
    """
    def __init__(self, energies_keV, n_channels=1):
        super().__init__()
        # D per channel: [n_channels, n_energies]
        D = torch.as_tensor(energies_keV, dtype=torch.float32).unsqueeze(0).expand(n_channels, -1)
        self.register_buffer("D", D)

    def forward(self, q):
        # q: [n_rays, n_exposures, n_energies]
        # D: [n_channels, n_energies]
        # We want to integrate along energies for each ray/exposure.
        # Output should be [n_rays, n_exposures, n_channels]
        # Using einsum: 'rec, kc -> rke' (incorrect)
        # We want 'rne, ce -> rnc' (n_rays, n_exposures, n_channels)
        return torch.einsum('rne,ce->rnc', q, self.D)
    
    def compute_forward_stats(self, q_detected_mean):
        """
        Compute y_mean and y_var from q_detected_mean.
        Assumes independent channels (variance = D^2 @ q_detected_mean).
        
        Args:
            q_detected_mean: [n_rays, n_exposures, n_energies] - mean detected spectral distribution
            
        Returns:
            y_mean: [n_rays, n_chan] - mean detector signal
            y_var: [n_rays, n_chan] - variance of detector signal
        """
        y_mean = self(q_detected_mean)  # [n_rays, n_exposures, n_chan]
        
        # Variance propagation: Var[y] = D^2 @ Var[q] = D^2 @ q_mean (Poisson)
        D_sq = self.D ** 2
        # Same einsum structure as forward
        y_var = torch.einsum('rne,ce->rnc', q_detected_mean, D_sq)  # [n_rays, n_exposures, n_chan]
        
        # Squeeze out exposures dimension to get [n_rays, n_chan]
        y_mean = y_mean.squeeze(1)
        y_var = y_var.squeeze(1)
        
        return y_mean, y_var

class DualExposureCTDetector(torch.nn.Module):
    """
    A detector specifically for dual-exposure data, where each exposure can have its own scaling.
    
    Each exposure maps to its own integration channel.
    Output Shape: [n_rays, n_channels=2]
    """
    def __init__(self, energies_keV):
        super().__init__()
        # We model this as each exposure having its own detector sensitivity D
        # n_channels = 1 for each of the 2 exposures externally.
        # But conceptually, we output [n_rays, 2] where 2 is the exposures.
        D = torch.as_tensor(energies_keV, dtype=torch.float32)
        # [2, E]
        self.register_buffer("D", D.unsqueeze(0).expand(2, -1).clone())

    def forward(self, q):
        # q: [n_rays, 2, n_energies]
        # D: [2, n_energies]
        # Each exposure has its own D channel. 
        # rnc = rne * ne -> rnc where n=c
        # We want y_r,n = sum_e q_r,n,e * D_n,e
        return torch.einsum('rne,ne->rn', q, self.D)
    
    def compute_forward_stats(self, q_detected_mean):
        """
        Compute y_mean and y_var from q_detected_mean.
        Assumes independent channels (variance = D^2 @ q_detected_mean).
        
        Args:
            q_detected_mean: [n_rays, n_exposures, n_energies] - mean detected spectral distribution
            
        Returns:
            y_mean: [n_rays, n_chan] - mean detector signal (n_chan==n_exposures==2)
            y_var: [n_rays, n_chan] - variance of detector signal
        """
        y_mean = self(q_detected_mean)  # [n_rays, n_chan] where n_chan==n_exposures
        
        # Variance propagation: Var[y] = D^2 @ Var[q] = D^2 @ q_mean (Poisson)
        D_sq = self.D ** 2
        # Same einsum structure as forward
        y_var = torch.einsum('rne,ne->rn', q_detected_mean, D_sq)  # [n_rays, n_chan]
        
        return y_mean, y_var

class EnergyIntegratingDetectorResponse(EnergyIntegratingCTDetector):
    """
    Legacy compatibility class.
    """
    pass
