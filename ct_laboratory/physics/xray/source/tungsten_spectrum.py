"""Tungsten-anode source spectrum  S_0  (paper notation:  S_0 = I_0 D{s_0(kVp)} D{exp(-Q_Al rho_Al l_Al)} ...).

    s(E; cone) = phi(E; kVp) * exp(-mu_Al(E) t_Al) * exp(-mu_W(E) t_W exp(k_heel * cone)) * exp(B(E) c)

phi(E; kVp)   spekpy unfiltered spectrum, linearly interpolated between tabulated tube voltages (photons per energy bin)
t_Al          aluminium-equivalent inherent + added filtration (mm)
t_W, k_heel   tungsten self-filtration (mm) and its exponential growth with cone angle (heel effect, 1/deg)
B c           smooth spectral correction, cubic B-splines on energy (n_spline coefficients)

Output shape: [..., n_energies] (broadcast over the shape of ``cone``), photons per bin, NOT normalized.
Per-source variation is handled by one instance per source (or by fitting t_Al / c per source with the shared prior).

Defaults reproduce the spectral-cylinder calibration (2026-10-01, 120 kV, e2e_lib): prior kVp = nominal +- 1, t_Al ~ 3 mm x/: 2,
t_W ~ 20 um x/: 5, spline +- 0.1, heel 0 +- 0.2.
"""
import math
import numpy as np
import torch
from scipy.interpolate import BSpline

from ct_laboratory.physics.xray.parameters import PriorParameters


def bspline_basis(energies_keV, n_spline=4, e_min=10.0, e_max=135.0):
    """[n_energies, n_spline] clamped cubic B-spline design matrix on [e_min, e_max] (n_spline >= 4)."""
    k = 3
    inner = np.linspace(e_min, e_max, n_spline - k + 1)[1:-1]
    knots = np.r_[[e_min] * (k + 1), inner, [e_max] * (k + 1)]
    e = np.clip(np.asarray(energies_keV, float), e_min, e_max - 1e-9)
    return BSpline.design_matrix(e, knots, k).toarray()


def spekpy_table(kvps, energy_edges_keV, th=12.0, anode='W'):
    """Unfiltered spekpy spectra integrated over energy bins: [len(kvps), n_bins] photons per bin (arbitrary scale)."""
    import spekpy
    edges = np.asarray(energy_edges_keV, float)
    out = []
    for kv in kvps:
        s = spekpy.Spek(kvp=float(kv), th=th, dk=0.25, targ=anode)
        k, f = s.get_spectrum()
        cum = np.r_[0, np.cumsum(f) * 0.25]
        kk = np.r_[k[0] - 0.125, k + 0.125]
        out.append(np.diff(np.interp(edges, kk, cum)))
    return np.asarray(out)


class TungstenSpectrum(torch.nn.Module):
    """Parameterized tungsten-anode spectrum with filtration, heel effect and a B-spline correction."""

    def __init__(self, energies_keV, kvps, phi, mu_al, mu_w, kvp=120.0, n_spline=4, priors=None):
        """
        energies_keV  [n_energies] bin centres
        kvps, phi     [n_kv], [n_kv, n_energies] unfiltered spekpy table (see spekpy_table)
        mu_al, mu_w   [n_energies] linear attenuation of Al and W (1/mm)
        priors        optional {name: (value, sd)} overrides of the default priors
        """
        super().__init__()
        f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        self.register_buffer('energies_keV', f32(energies_keV))
        self.register_buffer('kvps', f32(kvps))
        self.register_buffer('phi', f32(phi))
        self.register_buffer('mu_al', f32(mu_al))
        self.register_buffer('mu_w', f32(mu_w))
        self.register_buffer('B', f32(bspline_basis(energies_keV, n_spline)))
        p = PriorParameters()
        p.add('kvp', kvp, 1.0, label='kVp')
        p.add('al_mm', 3.0, math.log(2.0), 'exp', label='Al filtration (mm)')
        p.add('w_mm', 0.02, math.log(5.0), 'exp', label='W filtration (um)', display=lambda v: 1e3 * v)
        p.add('heel', 0.0, 0.2, label='heel slope (1/deg)')
        p.add('spline', 0.0, 0.1, shape=(n_spline,), label='spectrum spline')
        for name, (v, sd) in (priors or {}).items():
            p.set_prior(name, v, sd)
        self.params = p

    def unfiltered(self):
        """phi at the current kVp (linear interpolation in the table, differentiable in kVp): [n_energies]"""
        k = self.params['kvp'].clamp(self.kvps[0] + 1e-3, self.kvps[-1] - 1e-3)
        i = int(torch.searchsorted(self.kvps, k.detach()).clamp(1, len(self.kvps) - 1)) - 1
        f = (k - self.kvps[i]) / (self.kvps[i + 1] - self.kvps[i])
        return (1 - f) * self.phi[i] + f * self.phi[i + 1]

    def before_anode_filtration(self):
        """spectrum after Al and the spline but before the cone-dependent W term: [n_energies]"""
        return self.unfiltered() * torch.exp(-self.mu_al * self.params['al_mm']) * torch.exp(self.B @ self.params['spline'])

    def forward(self, cone_deg=None):
        """emitted fluence spectrum [..., n_energies]; ``cone_deg`` [...] = cone angle relative to the central plane"""
        s = self.before_anode_filtration()
        t_w = self.params['w_mm']
        if cone_deg is None:
            return s * torch.exp(-self.mu_w * t_w)
        t_w = t_w * torch.exp(self.params['heel'] * cone_deg)
        return s * torch.exp(-self.mu_w * t_w[..., None])
