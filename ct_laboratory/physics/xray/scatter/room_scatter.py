"""Room scatter  phi^R  - present in BOTH the gain (air) scan and the phantom scan.

Radiation scattered by the gantry, collimator, walls and table mount reaches every pixel as a smooth fluence.  Its
contribution is a fraction r of the MEAN air fluence F_bar of the source, so relative to the air fluence F_j of pixel j it is
r F_bar / F_j (the sharp detector efficiency eta_j cancels in t = y / I_0 because room scatter is detected through the same
S_1,j; the smooth beam pattern F_j does not).  Part h of it is shadowed by the object (it originates near the source and passes
through the object), modelled as a wide Gaussian blur of the primary transmission:

    phantom scan:  R_j = r (F_bar / F_j) [ (1 - h) + h (K_R T)_j ]
    gain scan:     R_j = r (F_bar / F_j)                                    (T = 1)

Because the gain scan also contains it, the air-normalized measurement is

    t_j = [ (1 - g - r) T_j + g G_j + R_j + O_j ] / [ (1 - g - r) + g + r F_bar / F_j ]

(see AirNormalizedProjectionModel).  Defaults: r = 4.5 % +- 3 % (spectral cylinders 120 kV), h ~ 0.2 (logit sd 1.5),
shadow kernel 220 mm source-plane sd (ACR module 1/3 fits).
"""
import math
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters
from ct_laboratory.physics.xray.scatter.kernels import blur_air_padded


class RoomScatter(torch.nn.Module):
    def __init__(self, fraction=0.045, fraction_sd=0.03, shadow=0.2, shadow_logit_sd=1.5, shadow_sd_mm=220.0):
        super().__init__()
        self.shadow_sd_mm = shadow_sd_mm
        p = PriorParameters()
        p.add('fraction', fraction, fraction_sd, label='room scatter (% of air)', display=lambda v: 100 * v)
        p.add('shadow', shadow, shadow_logit_sd, 'sigmoid', label='room scatter shadowed fraction')
        self.params = p

    @property
    def fraction(self):
        return self.params['fraction']

    def air(self, fluence_ratio):
        """gain-scan contribution r F_bar / F_j (fraction of the primary air signal)"""
        return self.fraction * fluence_ratio

    def forward(self, T, fluence_ratio, arc_mm, row_mm, magnification):
        """phantom-scan contribution R_j [..., n_rows, n_cols]"""
        h = self.params['shadow']
        sh = blur_air_padded(T, arc_mm, row_mm, torch.as_tensor(self.shadow_sd_mm * magnification, device=T.device))
        return self.fraction * fluence_ratio * ((1 - h) + h * sh)
