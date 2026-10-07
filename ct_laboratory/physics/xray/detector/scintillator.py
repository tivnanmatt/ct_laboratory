"""Indirect energy-integrating detector response  S_2 S_1  (paper notation).

    S_1,j = eta_j ( I - D{ exp(-mu_scint t_scint / cos theta_j) } )      interaction probability, oblique path 1/cos(incidence)
    S_2   = E^T / E_ref                                                   energy integration (signal proportional to deposited energy)

``detected_weight(cos_inc)`` returns the diagonal of S_2 S_1 per pixel, [..., n_energies].  The relative efficiency eta_j
(fill factor, scintillator non-uniformity) is sharp in j and CANCELS in air-normalized data t = y / I_0 for every
component whose incident fluence is proportional to the air fluence; it is therefore not a parameter here (see
AirNormalizedProjectionModel for the components where it does not cancel).

Default prior on the CsI thickness: 0.6 mm x/: exp(0.2) (spectral cylinders).
"""
import math
import numpy as np
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters


class ScintillatorDetector(torch.nn.Module):
    def __init__(self, energies_keV, mu_scint, thickness_mm=0.6, thickness_sd=0.2):
        """mu_scint [n_energies] linear attenuation of the scintillator (1/mm), e.g. CsI 4.51 g/cm^3"""
        super().__init__()
        self.register_buffer('energies_keV', torch.as_tensor(np.asarray(energies_keV), dtype=torch.float32))
        self.register_buffer('mu_scint', torch.as_tensor(np.asarray(mu_scint), dtype=torch.float32))
        self.params = PriorParameters().add('thickness_mm', thickness_mm, thickness_sd, 'exp', label='CsI thickness (mm)')

    def absorbed_fraction(self, cos_inc, energies_keV=None, mu_scint=None):
        """S_1 diagonal: 1 - exp(-mu t / cos)  [..., n_energies]"""
        mu = self.mu_scint if mu_scint is None else mu_scint
        return 1.0 - torch.exp(-mu * self.params['thickness_mm'] / cos_inc[..., None])

    def detected_weight(self, cos_inc, energies_keV=None, mu_scint=None):
        """S_2 S_1 diagonal: E (1 - exp(-mu t / cos))  [..., n_energies]; optionally at shifted energies (scatter)"""
        e = self.energies_keV if energies_keV is None else energies_keV
        return e * self.absorbed_fraction(cos_inc, mu_scint=mu_scint)
