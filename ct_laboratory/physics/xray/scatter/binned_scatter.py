"""Alternative scatter model: arbitrary LOW-FREQUENCY ADDITIVE signal, piecewise flat on 8 x 8 mm detector bins.

Instead of physical room / object scatter, the scatter is a free smooth additive signal, which may differ between the gain
(air) scan and the phantom scan:

    raw phantom signal   y_j  = I_0p,j T_j + s_ph,j
    raw gain signal      I_0j = I_0p,j     + s_g,j

With a_g = s_g / I_0 and a_ph = s_ph / I_0 (fractions of the MEASURED air signal) the air-normalized measurement is

    t_j = (1 - a_g,j) T_j + a_ph,j

where T_j is the scatter-free transmission (primary + off-focal).  The fields are parameterized on bins of B x B detector
pixels (B = 8 -> 8 x 8 mm on the 1 mm pitch), flat over each bin: a 32 x 32 module has 4 x 4 = 16 scatter points per field.
Parameters per source: a_g [n_bins] and a_ph [n_phantom_fields, n_bins] (e.g. one phantom field per station, or one shared
by stations that see the same object), with Gaussian priors (gain 4 % +- 3 %, phantom 4 % +- 5 % by default) and an optional
smoothness penalty  lambda * sum over neighbouring bins (a_b - a_b')^2  (neighbours within a module and across adjacent
modules along the arc).

Identifiability: in air (T = 1) only a_ph - a_g is determined; the split comes from the variation of T inside a bin and
across stations, plus the priors.  Use ``smoothness`` > 0 when a bin has few valid pixels.
"""
import torch

from ct_laboratory.physics.xray.parameters import PriorParameters


def module_bin_index(module, col, row, bin_px=8, module_cols=32, module_rows=32):
    """flat bin index per pixel from (module, col, row) indices: bins are bin_px x bin_px inside each module"""
    nbc, nbr = module_cols // bin_px, module_rows // bin_px
    return (module * nbc + col // bin_px) * nbr + row // bin_px


def neighbour_pairs(n_modules, bins_cols=4, bins_rows=4, module_order=None):
    """[n_pairs, 2] bin-index pairs adjacent in the detector plane (within modules, and across consecutive modules along the
    arc in ``module_order``: last bin column of one module next to the first of the next)."""
    pairs = []
    idx = lambda m, c, r: (m * bins_cols + c) * bins_rows + r
    for m in range(n_modules):
        for c in range(bins_cols):
            for r in range(bins_rows):
                if c + 1 < bins_cols: pairs.append((idx(m, c, r), idx(m, c + 1, r)))
                if r + 1 < bins_rows: pairs.append((idx(m, c, r), idx(m, c, r + 1)))
    order = list(range(n_modules)) if module_order is None else list(module_order)
    for m0, m1 in zip(order[:-1], order[1:]):
        for r in range(bins_rows):
            pairs.append((idx(m0, bins_cols - 1, r), idx(m1, 0, r)))
    return torch.tensor(pairs, dtype=torch.long)


class BinnedAdditiveScatter(torch.nn.Module):
    def __init__(self, bin_index, n_bins, n_phantom_fields=1, gain=0.04, gain_sd=0.03, phantom=0.04, phantom_sd=0.05,
                 smoothness=0.0, pairs=None):
        """bin_index [...] (pixel -> bin, e.g. module_bin_index); smoothness = lambda of the neighbour penalty"""
        super().__init__()
        self.register_buffer('bin_index', torch.as_tensor(bin_index, dtype=torch.long))
        self.register_buffer('pairs', pairs if pairs is not None else torch.zeros(0, 2, dtype=torch.long))
        self.n_bins, self.smoothness = n_bins, smoothness
        p = PriorParameters()
        p.add('gain', gain, gain_sd, shape=(n_bins,), label='gain-scan scatter (% of air)', display=lambda v: 100 * v)
        p.add('phantom', phantom, phantom_sd, shape=(n_phantom_fields, n_bins), label='phantom-scan scatter (% of air)', display=lambda v: 100 * v)
        self.params = p

    def gain_field(self):
        """a_g per pixel [...]"""
        return self.params['gain'][self.bin_index]

    def phantom_field(self, field=0):
        """a_ph per pixel [...]"""
        return self.params['phantom'][field][self.bin_index]

    def forward(self, T, field=0):
        """t = (1 - a_g) T + a_ph"""
        return (1 - self.gain_field()) * T + self.phantom_field(field)

    def penalty(self):
        if self.smoothness <= 0 or len(self.pairs) == 0:
            return 0.0
        g = self.params['gain']; ph = self.params['phantom']
        d = (g[self.pairs[:, 0]] - g[self.pairs[:, 1]]) ** 2
        dp = (ph[:, self.pairs[:, 0]] - ph[:, self.pairs[:, 1]]) ** 2
        return self.smoothness * (d.double().sum() + dp.double().sum())
