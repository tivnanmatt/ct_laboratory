"""Alternative scatter model: arbitrary LOW-FREQUENCY ADDITIVE signal, piecewise flat on 8 x 8 mm detector bins.

Instead of physical room / object scatter, the scatter is a free smooth additive signal, which may differ between the gain
(air) scan and the phantom scan:

    raw phantom signal   y_j  = I_0p,j T_j + s_ph,j
    raw gain signal      I_0j = I_0p,j     + s_g,j

With a_g = s_g / I_0 and a_ph = s_ph / I_0 (fractions of the MEASURED air signal) the air-normalized measurement is

    t_j = (1 - a_g,j) T_j + a_ph,j

where T_j is the scatter-free transmission (primary + off-focal).  The phantom field is parameterized on bins of B x B detector
pixels (B = 8 -> 8 x 8 mm on the 1 mm pitch), flat over each bin: a 32 x 32 module has 4 x 4 = 16 scatter points.

Gain-scan field (``gain_flat``):
    True   one flat level a_g per source (room scatter: there is no object in the gain scan, so it cannot have object structure)
    False  one value per bin, like the phantom field
Phantom-scan field (``phantom_relative_to_gain``):
    True   a_ph,b = a_g + delta_b,  delta_b ~ N(0, phantom_sd^2): the room-scatter level is the prior for the phantom scatter and
           the bins model the object-dependent change (object scatter minus shadowed room scatter)
    False  a_ph,b ~ N(phantom, phantom_sd^2) independently of a_g
Smoothness:
    neighbour penalty   lambda * sum over neighbouring bins (a_b - a_b')^2  (smoothness, pairs)
    Laplacian penalty   lambda_g ||Delta a_g||^2 + lambda_ph ||Delta a_ph||^2  on the bin GRID (grid [n_bin_rows, n_bin_cols] of bin
                        indices in detector order, e.g. bin_grid(...)); Delta = 3 x 3 Laplacian kernel, reflection-padded convolution, so a
                        constant or linear field costs nothing.  Use a very high lambda_g (gain scan: room scatter only, very low frequency)
                        and a lower lambda_ph (phantom scan: as smooth as the data allow).
    With phantom_relative_to_gain the phantom penalty acts on delta = a_ph - a_g.

Identifiability: in air (T = 1) only a_ph - a_g is determined; the split comes from the variation of T inside a bin and across
stations, plus the priors.  The fields are signal fractions at normal incidence; with a ScatterSpectrum the per-pixel value is
multiplied by the detection ratio of the scatter spectrum relative to the primary (see AirNormalizedProjectionModel).
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
                 smoothness=0.0, pairs=None, gain_flat=False, phantom_relative_to_gain=False, grid=None, laplacian_gain=0.0, laplacian_phantom=0.0):
        """bin_index [...] (pixel -> bin, e.g. module_bin_index); smoothness = lambda of the neighbour penalty"""
        super().__init__()
        self.register_buffer('bin_index', torch.as_tensor(bin_index, dtype=torch.long))
        self.register_buffer('pairs', pairs if pairs is not None else torch.zeros(0, 2, dtype=torch.long))
        self.n_bins, self.smoothness = n_bins, smoothness
        self.gain_flat, self.relative = gain_flat, phantom_relative_to_gain
        self.register_buffer('grid', torch.as_tensor(grid, dtype=torch.long) if grid is not None else torch.zeros(0, 0, dtype=torch.long))
        self.laplacian_gain, self.laplacian_phantom = laplacian_gain, laplacian_phantom
        p = PriorParameters()
        p.add('gain', gain, gain_sd, shape=() if gain_flat else (n_bins,),
              label='gain-scan scatter, flat (% of air)' if gain_flat else 'gain-scan scatter (% of air)', display=lambda v: 100 * v)
        if phantom_relative_to_gain:
            p.add('phantom', 0.0, phantom_sd, shape=(n_phantom_fields, n_bins), label='phantom − gain scatter (% of air)', display=lambda v: 100 * v)
        else:
            p.add('phantom', phantom, phantom_sd, shape=(n_phantom_fields, n_bins), label='phantom-scan scatter (% of air)', display=lambda v: 100 * v)
        self.params = p

    def gain_bins(self):
        g = self.params['gain']
        return g.expand(self.n_bins) if self.gain_flat else g

    def phantom_bins(self, field=0):
        ph = self.params['phantom'][field]
        return self.gain_bins() + ph if self.relative else ph

    def gain_field(self):
        """a_g per pixel [...]"""
        return self.gain_bins()[self.bin_index]

    def phantom_field(self, field=0):
        """a_ph per pixel [...]"""
        return self.phantom_bins(field)[self.bin_index]

    def forward(self, T, field=0):
        """t = (1 - a_g) T + a_ph"""
        return (1 - self.gain_field()) * T + self.phantom_field(field)

    def laplacian(self, values):
        """Laplacian of per-bin values [..., n_bins] on the bin grid (reflection padding): [..., n_bin_rows, n_bin_cols]"""
        img = values[..., self.grid]
        sh = img.shape; x = img.reshape(-1, 1, sh[-2], sh[-1])
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
        k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=values.device, dtype=values.dtype)[None, None]
        return torch.nn.functional.conv2d(x, k).reshape(sh)

    def penalty(self):
        tot = 0.0
        i, j = self.pairs[:, 0], self.pairs[:, 1]
        ph = self.params['phantom']
        if self.smoothness > 0 and len(self.pairs) > 0:
            tot = tot + self.smoothness * ((ph[:, i] - ph[:, j]) ** 2).double().sum()
            if not self.gain_flat:
                g = self.params['gain']; tot = tot + self.smoothness * ((g[i] - g[j]) ** 2).double().sum()
        if self.grid.numel() > 0:
            if self.laplacian_phantom > 0:
                tot = tot + self.laplacian_phantom * (self.laplacian(ph) ** 2).double().sum()
            if self.laplacian_gain > 0 and not self.gain_flat:
                tot = tot + self.laplacian_gain * (self.laplacian(self.params['gain']) ** 2).double().sum()
        return tot


def bin_grid(bin_index, arc_position, bins_rows=4, bin_row_index=None):
    """[bins_rows, n_bins / bins_rows] grid of bin indices in detector order: rows = bin row, columns sorted by the mean arc
    position of each bin's pixels.  bin_index, arc_position [...] per pixel; bin_row_index [...] per pixel (row // bin_px)."""
    bi = bin_index.flatten(); ap = arc_position.flatten().float(); br = bin_row_index.flatten()
    n = int(bi.max()) + 1
    pos = torch.zeros(n, device=bi.device).index_add_(0, bi, ap) / torch.bincount(bi, minlength=n).clamp(min=1)
    row = torch.zeros(n, dtype=torch.long, device=bi.device); row[bi] = br
    rows = [torch.nonzero(row == r)[:, 0] for r in range(bins_rows)]
    return torch.stack([b[torch.argsort(pos[b])] for b in rows])
