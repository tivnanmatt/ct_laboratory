"""Object scatter  phi^O  - single-scatter Klein-Nishina model with ONE scatter point at the object centroid.

All object scatter is lumped into a point c (the centroid) holding n_e electrons.  For source a, detector pixel j:

    O_j / air_j = alpha * sum_E s(E) F_j(E) / sum_E s(E) D_air,j(E)

    F_j(E)    = n_e  (dsigma_KN/dOmega)(E, theta_j)                         Klein-Nishina, per electron (mm^2 / sr)
                * exp(-mu(E) l_in)                                           attenuation on the way in
                * exp(-mu(E'_j) l_out)                                       attenuation on the way out, at the Compton-shifted energy E'
                * E'_j (1 - exp(-mu_scint(E'_j) t / cos theta_o))            detection of the scattered photon (S_2 S_1 at E')
                * (cos theta_o / d_oj^2) / (cos theta_a / d_aj^2) / d_ac^2   solid angles relative to the air signal
    D_air,j(E) = E (1 - exp(-mu_scint(E) t / cos theta_a))                   detection of the primary air beam

E'/E = 1 / (1 + (E / 511 keV)(1 - cos theta)), theta = angle between (c - a) and (j - c).  alpha (source dependent; prior 1 x/: 3)
absorbs the error of the one-point approximation (multiple scatter, spatial extent).  The geometric factors depend on the
object pose but not on the parameters, so they are precomputed once per (source, station) with ``geometry_factors``.

The result is in units of the PRIMARY air signal, i.e. it is added to the numerator of t (AirNormalizedProjectionModel).
For a water cylinder of radius R and effective height H0 use  c = axis point at the slab centre, l_in = l_out = R,
n_e = 3.343e20 / mm^3 * rho_e * pi R^2 H0  (H0 = 30 mm in the spectral-cylinder calibration).
"""
import math
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters

R_E2_HALF = 0.5 * 7.9408e-24          # r_e^2 / 2 in mm^2
WATER_ELECTRONS_PER_MM3 = 3.343e20


def interp1(x, xp, fp):
    """torch linear interpolation of a tabulated function (xp increasing), clamped at the ends"""
    i = torch.searchsorted(xp, x.clamp(xp[0], xp[-1])).clamp(1, len(xp) - 1)
    x0, x1 = xp[i - 1], xp[i]
    f = (x.clamp(xp[0], xp[-1]) - x0) / (x1 - x0)
    return fp[i - 1] * (1 - f) + fp[i] * f


def klein_nishina(E_keV, cos_theta):
    """(dsigma/dOmega per electron in mm^2/sr, scattered energy E' in keV), broadcast over E and cos_theta"""
    P = 1.0 / (1.0 + E_keV / 511.0 * (1.0 - cos_theta))
    return R_E2_HALF * P ** 2 * (P + 1.0 / P - (1.0 - cos_theta ** 2)), E_keV * P


def geometry_factors(source, point, det_pos, det_normal, energies_keV, fine_keV, mu_obj_fine, mu_scint_fine,
                     l_in, l_out, n_electrons, scint_mm=0.6):
    """Precompute (F [..., n_E], D_air [..., n_E]) for one source position.

    source [3], point [3] (scatter centroid), det_pos/det_normal [..., 3]; fine_keV/mu_*_fine: fine tabulation used at the
    shifted energies; mu_obj_fine is the object's linear attenuation (1/mm); l_in/l_out in mm.
    """
    E = energies_keV
    din = point - source; d_ac = din.norm(); din = din / d_ac
    do = det_pos - point; d_oj = do.norm(dim=-1); do = do / d_oj[..., None]
    da = det_pos - source; d_aj = da.norm(dim=-1)
    cos_t = (do * din).sum(-1)
    cos_o = (do * det_normal).sum(-1).abs()
    cos_a = (da * det_normal).sum(-1).abs() / d_aj
    kn, Es = klein_nishina(E, cos_t[..., None])
    mu_in = interp1(E, fine_keV, mu_obj_fine)
    mu_out = interp1(Es, fine_keV, mu_obj_fine)
    det = Es * (1 - torch.exp(-interp1(Es, fine_keV, mu_scint_fine) * scint_mm / cos_o[..., None]))
    solid = (cos_o / d_oj ** 2) / (cos_a / d_aj ** 2) / d_ac ** 2
    F = n_electrons * kn * torch.exp(-mu_in * l_in) * torch.exp(-mu_out * l_out) * det * solid[..., None]
    D_air = E * (1 - torch.exp(-interp1(E, fine_keV, mu_scint_fine) * scint_mm / cos_a[..., None]))
    return F, D_air


class CentroidKleinNishinaScatter(torch.nn.Module):
    def __init__(self, alpha=1.0, alpha_logsd=math.log(3.0)):
        super().__init__()
        self.params = PriorParameters().add('alpha', alpha, alpha_logsd, 'exp', label='object scatter (x physical)')

    def forward(self, spectrum, factors, density_scale=1.0):
        """O_j / air_j [...] for source spectrum [n_E] (before the detector) and precomputed (F, D_air)"""
        F, D_air = factors
        return self.params['alpha'] * density_scale * (F * spectrum).sum(-1) / (D_air * spectrum).sum(-1)
