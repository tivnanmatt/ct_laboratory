"""Checks for the scatter / off-focal / spectrum modules and the air-normalized projection model.

1. air (l = 0): t == 1 for the physical model (room scatter in gain AND phantom scan) for any parameters / beam pattern
2. binned model in air: t == 1 - a_g + a_ph, and 16 bins per 32 x 32 module
3. gradients reach every free parameter of both models
4. cylinder chords: centre chord = 2R, outside = 0, slab clipping
5. Klein-Nishina: total cross-section integrates to the Thomson value at low energy; one-point object scatter for a 30 mm
   water cylinder is O(0.01-0.1 %) of air (spectral-cylinder calibration: ~0.03 %)
"""
import math
import numpy as np
import torch
import xraydb

from ct_laboratory.physics.xray.source import TungstenSpectrum, spekpy_table
from ct_laboratory.physics.xray.detector import ScintillatorDetector
from ct_laboratory.physics.xray.scatter import (OffFocalRadiation, FocalSpotBlur, RoomScatter, CentroidKleinNishinaScatter,
                                                BinnedAdditiveScatter, module_bin_index, neighbour_pairs, geometry_factors,
                                                klein_nishina, WATER_ELECTRONS_PER_MM3)
from ct_laboratory.physics.xray.xray_system import AirNormalizedProjectionModel
from ct_laboratory.tomography.analytic_cylinder import cylinder_chords

torch.manual_seed(0)
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
edges = np.arange(10.0, 137.0, 2.0); E = 0.5 * (edges[1:] + edges[:-1]); keV = E * 1e3
mu_w = xraydb.material_mu('water', keV) / 10; mu_al = xraydb.material_mu('Al', keV) / 10; mu_W = xraydb.mu_elam('W', keV) * 19.3 / 10
csi = lambda k: (0.5115 * xraydb.mu_elam('Cs', k) + 0.4885 * xraydb.mu_elam('I', k)) * 4.51 / 10
kvps = np.arange(110.0, 131.0, 5.0)
phi = spekpy_table(kvps, edges)
T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)

src = TungstenSpectrum(E, kvps, phi, mu_al, mu_W, kvp=120.0).to(dev)
det = ScintillatorDetector(E, csi(keV)).to(dev)
nm, R_, C_ = 3, 32, 96
arc = T(np.arange(C_) * 1.0 + np.repeat(np.arange(nm) * 2.0, 32)); rows = T(np.arange(R_) - 15.5)
cos_inc = 0.8 + 0.2 * torch.rand(R_, C_, device=dev); cone = torch.randn(R_, C_, device=dev)
fr = 0.8 + 0.4 * torch.rand(R_, C_, device=dev)
ok = True


def check(name, cond, detail=''):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1 physical model in air, random parameters
phys = AirNormalizedProjectionModel(src, det, OffFocalRadiation(), room=RoomScatter(), object_scatter=None, focal_blur=FocalSpotBlur()).to(dev)
for p in phys.parameter_sets().values():
    with torch.no_grad():
        for z in p.z.values(): z.normal_()
Ql0 = torch.zeros(R_, C_, 4, len(E), device=dev)
o = phys(Ql0, cos_inc, cone, arc, rows, 0.74, fluence_ratio=fr)
check('physical model: t == 1 in air', (o['t'] - 1).abs().max() < 1e-5, f"max |t-1| = {float((o['t'] - 1).abs().max()):.1e}")

# 2 binned model
mod = torch.arange(nm, device=dev)[None, :, None].expand(R_, nm, 32).reshape(R_, C_)
col = torch.arange(32, device=dev)[None, None, :].expand(R_, nm, 32).reshape(R_, C_)
row = torch.arange(R_, device=dev)[:, None].expand(R_, C_)
bi = module_bin_index(mod, col, row)
check('binned: 16 bins per module, 64 px per bin', len(bi.unique()) == 16 * nm and bool((torch.bincount(bi.flatten()) == 64).all()))
pairs = neighbour_pairs(nm).to(dev)
check('binned: neighbour pairs', len(pairs) == nm * 24 + (nm - 1) * 4, f"{len(pairs)} pairs")
bs = BinnedAdditiveScatter(bi, 16 * nm, smoothness=10.0, pairs=pairs).to(dev)
with torch.no_grad():
    bs.params.z['gain'].normal_(); bs.params.z['phantom'].normal_()
binm = AirNormalizedProjectionModel(src, det, OffFocalRadiation(), binned=bs).to(dev)
o = binm(Ql0, cos_inc, cone, arc, rows, 0.74)
check('binned: t == 1 - a_g + a_ph in air', (o['t'] - (1 - bs.gain_field() + bs.phantom_field())).abs().max() < 1e-5)

# 4 cylinder chords
bs_ = (0.0, 50.0, 99.0, 100.5, 0.0)
s3 = T([[-500.0, b, 0.0] for b in bs_]); d3 = T([[500.0, b, 0.0] for b in bs_])
s3[4, 2] = -20.0; d3[4, 2] = 20.0                                                  # tilted ray: half of its chord has z > 0
Lfull, Lslab = cylinder_chords(s3, d3, T(0.0), T(0.0), T(100.0), z_slabs=((-1e4, 1e4), (0.0, 1e4)))
check('cylinder chords', abs(float(Lfull[0]) - 200) < 1e-3 and abs(float(Lfull[1]) - 2 * math.sqrt(100 ** 2 - 50 ** 2)) < 1e-2 and float(Lfull[3]) == 0.0
      and abs(float(Lslab[4]) - float(Lfull[4]) / 2) < 1e-2, f"{[round(float(v), 2) for v in Lfull]}, slab {float(Lslab[4]):.2f}")

# 3 gradients (non-air object: 150 mm water through the middle columns)
L = torch.zeros(R_, C_, 4, device=dev); L[:, 30:66] = 150.0
Ql = L[..., None] * T(mu_w)
for name, m in (('physical', phys), ('binned', binm)):
    for p in m.parameter_sets().values(): p.free()
    o = m(Ql, cos_inc, cone, arc, rows, 0.74, fluence_ratio=fr)
    (o['t'].sum() + m.neg_log_prior()).backward()
    bad = [f'{c}.{n}' for c, p in m.parameter_sets().items() for n, z in p.z.items() if z.grad is None or not torch.isfinite(z.grad).all() or float(z.grad.abs().sum()) == 0]
    check(f'{name}: gradients reach all parameters', not bad, str(bad))

# 5 Klein-Nishina + one-point object scatter
ct = torch.linspace(-1, 1, 20001, device=dev, dtype=torch.float64)
kn, _ = klein_nishina(torch.tensor(1e-3, device=dev, dtype=torch.float64), ct)
sig = float(2 * math.pi * torch.trapz(kn, ct)); thomson = 8 * math.pi / 3 * 2 * 0.5 * 7.9408e-24
check('KN -> Thomson at low energy', abs(sig / thomson - 1) < 1e-3, f"ratio {sig / thomson:.5f}")
fine = np.arange(5.0, 140.0, 0.25)
source = T([-600.0, 0.0, 0.0]); point = T([0.0, 0.0, 0.0])
ang = torch.linspace(-0.6, 0.6, 50, device=dev)
pos = torch.stack([400 * torch.cos(ang), 400 * torch.sin(ang), torch.zeros_like(ang)], -1); nrm = pos / pos.norm(dim=-1, keepdim=True)
F, Dair = geometry_factors(source, point, pos, nrm, T(E), T(fine), T(xraydb.material_mu('water', fine * 1e3) / 10), T(csi(fine * 1e3)),
                           15.0, 15.0, WATER_ELECTRONS_PER_MM3 * math.pi * 15.0 ** 2 * 30.0)
O = CentroidKleinNishinaScatter().to(dev)(src(), (F, Dair))
check('one-point object scatter of a 30 mm water cylinder ~ 0.01-0.1 % of air, symmetric', 1e-4 < float(O.mean()) < 1e-3 and float((O - O.flip(0)).abs().max() / O.max()) < 1e-3,
      f"mean {100 * float(O.mean()):.3f} % of air")
print('ALL PASS' if ok else 'SOME CHECKS FAILED')
