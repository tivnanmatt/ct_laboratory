"""Off-focal (extra-focal / gantry) radiation  phi^G.

X-rays emitted from a wide halo around the focal spot (stem / housing backscatter), same spectrum as the primary.  In
air-normalized units

    G_j = sum_j' [K_G]_jj' T_j'          T = primary transmission of the detected spectrum
    t   = (1 - g - ...) T + g G + ...

K_G is a Gaussian whose source-plane sd is sqrt(sd_focal^2 + sd_halo^2), projected to the detector through the object
(magnification m = (d_sd - d_so) / d_so).  It is present in the gain scan too, where G = 1, so the fraction g cancels
from the air normalization.

Defaults: g = 10 % +- 5 %, halo sd 50 mm x/: 2 (priors of the spectral-cylinder fit, which found 12.9 % and 61 mm at 120 kV);
focal spot 1 mm FWHM.
"""
import math
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters
from ct_laboratory.physics.xray.scatter.kernels import blur_air_padded


class OffFocalRadiation(torch.nn.Module):
    def __init__(self, fraction=0.10, fraction_sd=0.05, halo_sd_mm=50.0, halo_sd_logsd=math.log(2.0), focal_fwhm_mm=1.0):
        super().__init__()
        self.focal_sd_mm = focal_fwhm_mm / 2.3548
        p = PriorParameters()
        p.add('fraction', fraction, fraction_sd, label='off-focal fraction (%)', display=lambda v: 100 * v)
        p.add('halo_sd_mm', halo_sd_mm, halo_sd_logsd, 'exp', label='off-focal halo sd (mm, source plane)')
        self.params = p

    @property
    def fraction(self):
        return self.params['fraction']

    def detector_sd(self, magnification):
        return torch.sqrt(self.focal_sd_mm ** 2 + self.params['halo_sd_mm'] ** 2) * magnification

    def forward(self, T, arc_mm, row_mm, magnification):
        """G [..., n_rows, n_cols] (air-normalized, not yet multiplied by the fraction)"""
        return blur_air_padded(T, arc_mm, row_mm, self.detector_sd(magnification))


class FocalSpotBlur(torch.nn.Module):
    """Primary blur by the focal spot (isotropic Gaussian, FWHM in the source plane), same boundary convention."""

    def __init__(self, focal_fwhm_mm=1.0):
        super().__init__()
        self.focal_sd_mm = focal_fwhm_mm / 2.3548

    def forward(self, T, arc_mm, row_mm, magnification):
        return blur_air_padded(T, arc_mm, row_mm, torch.as_tensor(self.focal_sd_mm * magnification, device=T.device))
