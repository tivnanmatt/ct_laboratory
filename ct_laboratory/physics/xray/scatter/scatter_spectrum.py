"""Spectral shape of scatter: a smooth (B-spline) distortion of the primary air spectrum.

    s_sc(E) = s_0(E) exp(B(E) c_sc)          c_sc ~ N(0, sd^2)   (sd 0.5 by default; c_sc = 0 -> scatter has the primary spectrum)

Scatter is Compton-shifted (softer) and filtered differently from the primary, so it is detected with a different efficiency.
For an energy-integrating detector the spectrum enters the air-normalized signal only through the per-pixel detection ratio

    rho_j = [ sum_E s_sc S_2 S_1,j / sum_E s_sc S_2 S_1,0 ] / [ sum_E s_0 S_2 S_1,j / sum_E s_0 S_2 S_1,0 ]

(S_1,j at the pixel's incidence angle, S_1,0 at normal incidence): a scatter field defined as a fraction of the air signal at normal
incidence becomes rho_j times that at pixel j.  rho_j departs from 1 only through the 1/cos theta path in the scintillator, so c_sc is
WEAKLY identified by energy-integrating data; its prior keeps it near the primary shape.
"""
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters


class ScatterSpectrum(torch.nn.Module):
    def __init__(self, n_spline=4, sd=0.5):
        super().__init__()
        self.params = PriorParameters().add('spline', 0.0, sd, shape=(n_spline,), label='scatter spectrum spline')

    def spectrum(self, source):
        """scatter spectrum [n_E] for a TungstenSpectrum ``source`` (its B-spline basis is reused)"""
        return source() * torch.exp(source.B @ self.params['spline'])

    def detection_ratio(self, source, detector, cos_inc):
        """rho_j [...] (1 at normal incidence)"""
        s0 = source(); ss = self.spectrum(source)
        Dj = detector.detected_weight(cos_inc); D0 = detector.detected_weight(torch.ones_like(cos_inc[:1, :1]))[0, 0]
        return ((Dj * ss).sum(-1) / (D0 * ss).sum()) / ((Dj * s0).sum(-1) / (D0 * s0).sum())
