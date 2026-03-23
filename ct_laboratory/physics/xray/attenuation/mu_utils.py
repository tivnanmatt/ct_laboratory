import torch
import xraydb
class Attenuator(torch.nn.Module):
    """
    A base module for attenuation of an input spectrum.
    
    Output Shape: [n_rays, n_exposures, n_energies]
    """
    def __init__(self):
        super().__init__()

    def forward(self, q0, transmission):
        # q0: [n_rays, n_exposures, n_energies]
        # transmission: [n_rays, n_exposures, n_energies] or similar broadcastable.
        return q0 * transmission
def get_mu(comp, energy_ev, density=1.0, kind='total'):
    """Helper to get attenuation coefficient in cm^-1, then convert to mm^-1."""
    if isinstance(comp, str):
        return xraydb.material_mu(comp, energy_ev, density=density) / 10.0
    else:
        mu_rho = 0.0
        for symbol, fraction in comp.items():
            mu_rho += fraction * xraydb.mu_elam(symbol, energy_ev, kind=kind)
        return (mu_rho * density) / 10.0
