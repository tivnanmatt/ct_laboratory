#!/usr/bin/env python
"""Test the voxel-driven separable-footprint projector (VoxelProjector3D).

Checks, in order:
  1. class hierarchy: RayProjector3D / VoxelProjector3D are both Projector3D
  2. footprint normalisation on a single voxel (analytic + vs 16-subray Siddon)
  3. adjointness  <A x, y> == <x, A^T y>   (on-the-fly and precomputed CSR)
  4. CSR footprints reproduce the on-the-fly kernels bit-for-bit (up to atomics)
  5. autograd backward == back_project
  6. pixel valid-mask handling
  7. sphere phantom: SF vs ray-driven Siddon with 1 ray/pixel and 4x4 sub-rays/pixel
  8. uniform volume: SF vs Siddon
  9. timing on a larger geometry (on-the-fly / CSR / ray-driven)
 10. CGLS reconstruction of the sphere with the voxel projector

Geometry is the library's own static-CT ring (build_uniform_static_3d_geometry),
so panel conventions are exactly those used elsewhere in ct_laboratory.
"""
import os, sys, time, math
import torch
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ct_laboratory.tomography import (
    Projector3D, RayProjector3D, CTProjector3DModule, VoxelProjector3D,
    views_from_module_orientations, precompute_tvals_stitched,
)
from ct_laboratory.tomography.staticct_projector_3d import build_uniform_static_3d_geometry

dev = torch.device("cuda")
here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "test_outputs"); os.makedirs(out_dir, exist_ok=True)
LOG = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)
def sync(): torch.cuda.synchronize()
def status(ok): return "PASS" if ok else "FAIL"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def grid_affine(n_x, n_y, n_z, vox):
    M = torch.eye(3) * vox
    b = torch.tensor([-(n_x - 1) / 2 * vox, -(n_y - 1) / 2 * vox, -(n_z - 1) / 2 * vox])
    return M.to(dev), b.to(dev)

def soft_sphere(n_x, n_y, n_z, vox, radius, edge=4.0, value=1.0):
    x = (torch.arange(n_x) - (n_x - 1) / 2) * vox
    y = (torch.arange(n_y) - (n_y - 1) / 2) * vox
    z = (torch.arange(n_z) - (n_z - 1) / 2) * vox
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    return (value * ((radius - r) / edge).clamp(0, 1)).to(torch.float32).contiguous()

def make_ray_projector(n_x, n_y, n_z, M, b, src, dst, compressed=False):
    """Ray-driven (Siddon) projector on the given rays with precomputed tvals.
    compressed=False keeps exact float32 tvals; True uses the uint16-delta path used in production."""
    tv = precompute_tvals_stitched(n_x, n_y, n_z, M, b, src, dst, chunk_size=200000,
                                   backend="cuda", device=dev, verbose=False, use_compression=compressed)
    return RayProjector3D(n_x, n_y, n_z, M, b, src, dst, backend="cuda", device=dev,
                          precomputed_intersections=True, tvals=tv, use_compression=compressed)

def subray_dst(vproj, n_sub):
    """dst for an n_sub x n_sub grid of sub-rays per pixel: [n_sub*n_sub, n_pix, 3]."""
    v = vproj.views
    offs = (torch.arange(n_sub, device=dev, dtype=torch.float32) + 0.5) / n_sub - 0.5
    OU, OV = torch.meshgrid(offs, offs, indexing="ij"); OU, OV = OU.reshape(-1), OV.reshape(-1)
    outs = []
    for w in range(vproj.n_view):
        r = v[w]; C, U, V = r[3:6], r[6:9], r[9:12]
        du, dv, nu, nv = float(r[15]), float(r[16]), int(round(float(r[17]))), int(round(float(r[18])))
        iu = torch.arange(nu, device=dev, dtype=torch.float32) - 0.5 * (nu - 1)
        iv = torch.arange(nv, device=dev, dtype=torch.float32) - 0.5 * (nv - 1)
        IU, IV = torch.meshgrid(iu, iv, indexing="ij")
        base_u = IU.reshape(-1); base_v = IV.reshape(-1)                       # [n_pix_w]
        uu = (base_u[None, :] + OU[:, None]) * du                                # [n_sub^2, n_pix_w]
        vv = (base_v[None, :] + OV[:, None]) * dv
        outs.append(C[None, None, :] + uu[..., None] * U[None, None, :] + vv[..., None] * V[None, None, :])
    return torch.cat(outs, dim=1).contiguous()

def rel_rms(a, b, mask=None):
    if mask is not None: a, b = a[mask], b[mask]
    return float((a - b).norm() / b.norm().clamp_min(1e-12))

def build_geometry(n_source, n_module, modules_per_source, n_u, n_v, pitch_u, pitch_v,
                   source_radius=280.0, module_radius=220.0):
    S, C, R, mask = build_uniform_static_3d_geometry(
        n_source=n_source, source_radius=source_radius, source_z_offset=0.0,
        n_module=n_module, module_radius=module_radius, module_z_offset=0.0,
        modules_per_source=modules_per_source, device=dev)
    views, pairs = views_from_module_orientations(S, C, R, pitch_u, pitch_v, n_u, n_v, mask, device=dev)
    return S, C, R, mask, views, pairs

# ---------------------------------------------------------------------------
# small geometry used for the correctness tests
# ---------------------------------------------------------------------------
n_x, n_y, n_z, vox = 48, 48, 8, 2.5
n_source, n_module, mps = 16, 24, 5
n_u, n_v, pitch_u, pitch_v = 32, 16, 1.8, 2.0
M, b = grid_affine(n_x, n_y, n_z, vox)
S, C, R, mask, views, pairs = build_geometry(n_source, n_module, mps, n_u, n_v, pitch_u, pitch_v)

vproj = VoxelProjector3D(n_x, n_y, n_z, M, b, views, device=dev)
src, dst = vproj.rays()
rproj = make_ray_projector(n_x, n_y, n_z, M, b, src, dst)                    # exact float tvals
rproj_c = make_ray_projector(n_x, n_y, n_z, M, b, src, dst, compressed=True)  # production uint16 path

P("=" * 78)
P("VoxelProjector3D (separable footprints) test")
P("=" * 78)
P(f"volume {n_x}x{n_y}x{n_z} @ {vox} mm; {n_source} sources x {mps} panels x {n_u}x{n_v} px "
  f"({pitch_u}x{pitch_v} mm) -> {vproj.n_view} views, {vproj.n_pix:,} pixels")
P(f"voxel projector: {vproj}")
P(f"ray   projector: {rproj}")

# ---------------------------------------------------------------- 1. hierarchy
P("\n[1] class hierarchy")
ok1 = (isinstance(vproj, Projector3D) and isinstance(rproj, Projector3D)
       and RayProjector3D is CTProjector3DModule and vproj.kind == "voxel" and rproj.kind == "ray"
       and vproj.n_ray == rproj.n_ray == src.shape[0])
P(f"  RayProjector3D is CTProjector3DModule: {RayProjector3D is CTProjector3DModule}")
P(f"  kinds: voxel='{vproj.kind}', ray='{rproj.kind}';  n_ray {vproj.n_ray} == {rproj.n_ray}   {status(ok1)}")

# ---------------------------------------------------- 2. single-voxel normalisation
P("\n[2] single-voxel footprint normalisation")
vol1 = torch.zeros(n_x, n_y, n_z, device=dev); i0, j0, k0 = n_x // 2, n_y // 2, n_z // 2; vol1[i0, j0, k0] = 1.0
s_sf = vproj.forward_project(vol1)
n_sub = 4
dst_sub = subray_dst(vproj, n_sub)
rproj_sub = make_ray_projector(n_x, n_y, n_z, M, b, src.repeat(n_sub * n_sub, 1), dst_sub.reshape(-1, 3))
def ray_subray(vol):
    return rproj_sub.forward_project(vol).reshape(n_sub * n_sub, -1).mean(0)
s_ray16 = ray_subray(vol1)
s_ray1 = rproj.forward_project(vol1)
# analytic: for one view, sum_pix w = V m^2 / (du dv cos(incidence))
c = M @ torch.tensor([i0, j0, k0], dtype=torch.float32, device=dev) + b
w_sel = int(torch.nonzero((pairs[:, 0] == 0) & (pairs[:, 1] == n_module // 2))[0])   # source 0, module across
r = views[w_sel]; Sv, Cv, nv_ = r[0:3], r[3:6], r[12:15]
d = (c - Sv); dist = d.norm(); d = d / dist
t = ((Cv - Sv) @ nv_) / (d @ nv_); m = float(t / dist)                                  # magnification
cos_inc = float(abs(d @ nv_))
expect = (vox ** 3) * m * m / (pitch_u * pitch_v) / cos_inc
sv = vproj.sinogram_view(s_sf)[w_sel].sum().item()
P(f"  view (src 0, module {n_module//2}): sum_pix w  SF = {sv:.4f}   analytic V m^2/(du dv cos) = {expect:.4f}   "
  f"ratio {sv/expect:.4f}")
tot_sf, tot_16, tot_1 = s_sf.sum().item(), s_ray16.sum().item(), s_ray1.sum().item()
P(f"  all views: sum SF = {tot_sf:.3f}   sum Siddon 16-subray = {tot_16:.3f} (ratio {tot_sf/tot_16:.4f})   "
  f"sum Siddon 1-ray = {tot_1:.3f} (ratio {tot_sf/tot_1:.4f})")
ok2 = abs(sv / expect - 1) < 0.02 and abs(tot_sf / tot_16 - 1) < 0.02
P(f"  {status(ok2)} (within 2%)")

# ------------------------------------------------------------------ 3. adjoint
P("\n[3] adjoint test  <A x, y>  vs  <x, A^T y>")
torch.manual_seed(0)
x = torch.rand(n_x, n_y, n_z, device=dev); y = torch.rand(vproj.n_pix, device=dev)
def adjoint_err(proj, x=x, y=y):
    lhs = (proj.forward_project(x) * y).sum().item(); rhs = (x * proj.back_project(y)).sum().item()
    return lhs, rhs, abs(lhs - rhs) / abs(lhs)
lhs, rhs, e_otf = adjoint_err(vproj)
P(f"  on-the-fly : <Ax,y> = {lhs:.6f}   <x,A^T y> = {rhs:.6f}   rel err {e_otf:.2e}")
nnz = vproj.precompute_footprints()
P(f"  precomputed footprints: nnz = {nnz:,}  ({vproj.footprint_bytes/1e6:.1f} MB, "
  f"{nnz/vproj.n_voxel:.1f} pixels per voxel, {nnz/vproj.n_pix:.1f} voxels per pixel)")
lhs, rhs, e_csr = adjoint_err(vproj)
P(f"  CSR        : <Ax,y> = {lhs:.6f}   <x,A^T y> = {rhs:.6f}   rel err {e_csr:.2e}")
lhs, rhs, e_ray = adjoint_err(rproj)
P(f"  ray-driven : <Ax,y> = {lhs:.6f}   <x,A^T y> = {rhs:.6f}   rel err {e_ray:.2e}   (reference)")
ok3 = e_otf < 1e-4 and e_csr < 1e-4
P(f"  {status(ok3)} (rel err < 1e-4)")

# ----------------------------------------------------------- 4. CSR == on-the-fly
P("\n[4] CSR footprints vs on-the-fly kernels")
s_csr = vproj.forward_project(x); v_csr = vproj.back_project(y)
vproj.clear_footprints()
s_otf = vproj.forward_project(x); v_otf = vproj.back_project(y)
d_f = (s_csr - s_otf).abs().max().item() / s_otf.abs().max().item()
d_b = (v_csr - v_otf).abs().max().item() / v_otf.abs().max().item()
P(f"  forward max|diff|/max = {d_f:.2e}   back max|diff|/max = {d_b:.2e}")
ok4 = d_f < 1e-4 and d_b < 1e-6
P(f"  {status(ok4)}")
vproj.precompute_footprints()

# ------------------------------------------------------------------ 5. autograd
P("\n[5] autograd: d/dx <A x, y>  ==  A^T y")
xg = x.clone().requires_grad_(True)
(vproj(xg) * y).sum().backward()
e_ag = (xg.grad - vproj.back_project(y)).abs().max().item() / vproj.back_project(y).abs().max().item()
P(f"  max|grad - A^T y| / max = {e_ag:.2e}   {status(e_ag < 1e-6)}")
ok5 = e_ag < 1e-6

# ---------------------------------------------------------------- 6. valid mask
P("\n[6] pixel valid mask")
valid = torch.ones(vproj.n_view, n_u, n_v, dtype=torch.bool, device=dev); valid[3] = False; valid[7, :, :4] = False
vmask = VoxelProjector3D(n_x, n_y, n_z, M, b, views, valid=valid, device=dev, precompute_footprints=True)
sm = vmask.sinogram_view(vmask.forward_project(x))
zero_ok = (sm[3].abs().max().item() == 0.0) and (sm[7, :, :4].abs().max().item() == 0.0)
lhs, rhs, e_m = adjoint_err(vmask)
P(f"  masked view 3 and rows 0-3 of view 7 are exactly zero: {zero_ok};  adjoint rel err {e_m:.2e}   "
  f"nnz {vmask.nnz:,} (vs {vproj.nnz:,} unmasked)")
ok6 = zero_ok and e_m < 1e-4
P(f"  {status(ok6)}")

# -------------------------------------------------------------- 7. sphere phantom
P("\n[7] soft sphere phantom: SF vs ray-driven Siddon")
sph = soft_sphere(n_x, n_y, n_z, vox, radius=38.0, edge=5.0).to(dev)
s_sf = vproj.forward_project(sph); s_1 = rproj.forward_project(sph); s_16 = ray_subray(sph); s_1c = rproj_c.forward_project(sph)
sup = s_16 > 0.05 * s_16.max()
P(f"  line integrals: max SF {s_sf.max():.3f}  max Siddon {s_1.max():.3f} mm")
P(f"  SF vs Siddon 1 ray/pixel   : rel RMS {rel_rms(s_sf, s_1, sup):.4f}   max|diff| {(s_sf-s_1).abs().max():.4f} mm")
P(f"  SF vs Siddon 16 subrays/px : rel RMS {rel_rms(s_sf, s_16, sup):.4f}   max|diff| {(s_sf-s_16).abs().max():.4f} mm")
P(f"  (Siddon 1 ray vs 16 subrays: rel RMS {rel_rms(s_1, s_16, sup):.4f}  <- pixel-area vs centre-ray, ray model itself)")
P(f"  NOTE ray-driven Siddon with uint16-COMPRESSED tvals vs float tvals: rel RMS {rel_rms(s_1c, s_1, sup):.4f}   "
  f"max|diff| {(s_1c - s_1).abs().max():.4f} mm  <- pre-existing error of the compressed path on grid-axis-aligned rays")
ok7 = rel_rms(s_sf, s_16, sup) < 0.01
P(f"  {status(ok7)} (SF within 1% of the 16-subray reference)")
# 7b: SF-TR is an approximation whose error shrinks with voxel size -> check it converges
P("  [7b] convergence with voxel size (same geometry, same sphere):")
for f_n, f_vox in [(48, 2.5), (96, 1.25)]:
    f_nz = n_z * int(round(vox / f_vox)); fM, fb = grid_affine(f_n, f_n, f_nz, f_vox)
    fv = VoxelProjector3D(f_n, f_n, f_nz, fM, fb, views, device=dev)
    fsph = soft_sphere(f_n, f_n, f_nz, f_vox, radius=38.0, edge=5.0).to(dev)
    f_src, _ = fv.rays()
    fr16 = make_ray_projector(f_n, f_n, f_nz, fM, fb, f_src.repeat(n_sub * n_sub, 1), subray_dst(fv, n_sub).reshape(-1, 3))
    f_s16 = fr16.forward_project(fsph).reshape(n_sub * n_sub, -1).mean(0); f_ssf = fv.forward_project(fsph)
    f_sup = f_s16 > 0.05 * f_s16.max()
    P(f"     voxel {f_vox:.2f} mm ({f_n}^2x{f_nz}): SF vs 16-subray rel RMS {rel_rms(f_ssf, f_s16, f_sup):.4f}   max|diff| {(f_ssf - f_s16).abs().max():.3f} mm")
    del fr16, fv; torch.cuda.empty_cache()
# 4b: column cache (culled per-column view lists + end-slice trapezoids) must reproduce on-the-fly
vproj_col = VoxelProjector3D(n_x, n_y, n_z, M, b, views, device=dev, cache="column")
d_f = rel_rms(vproj_col.forward_project(sph), vproj.forward_project(sph)); d_b = rel_rms(vproj_col.back_project(s_16), vproj.back_project(s_16))
ok4b = d_f < 1e-4 and d_b < 1e-4
P(f"  [4b] column cache: entries {vproj_col.column_entries} ({vproj_col.column_bytes/1e6:.1f} MB); rel RMS vs on-the-fly fwd {d_f:.1e} back {d_b:.1e}  {status(ok4b)}")
# 7c: one row of voxels along x seen by the grid-aligned source 0 -> exact path length is known
row = torch.zeros(n_x, n_y, n_z, device=dev); row[:, n_y // 2, n_z // 2] = 1.0
w_c = int(torch.nonzero((pairs[:, 0] == 0) & (pairs[:, 1] == n_module // 2))[0])
a_sf = vproj.sinogram_view(vproj.forward_project(row))[w_c]; a_f = vproj.sinogram_view(rproj.forward_project(row))[w_c]
a_c = vproj.sinogram_view(rproj_c.forward_project(row))[w_c]; cc = int(a_sf[:, n_v // 2].argmax())
P(f"  [7c] row of {n_x} voxels along x, grid-aligned source 0, column {cc}: exact {n_x*vox:.2f} mm;  "
  f"SF {float(a_sf[cc, n_v//2]):.2f}   Siddon float {float(a_f[cc, n_v//2]):.2f}   Siddon compressed {float(a_c[cc, n_v//2]):.2f}")
ok7 = ok7 and abs(float(a_sf[cc, n_v//2]) - n_x * vox) < 0.05
# figure: source 0, its 5 panels side by side along u
fig, ax = plt.subplots(4, 1, figsize=(16, 9), constrained_layout=True)
sel = torch.nonzero(pairs[:, 0] == 0).flatten()
def strip(s):  # [n_pix] -> [n_v, mps*n_u]
    return torch.cat([vproj.sinogram_view(s)[w].T for w in sel], dim=1).cpu().numpy()
vmax = strip(s_16).max()
for a, (img, t) in zip(ax, [(strip(s_sf), "voxel-driven separable footprints"),
                            (strip(s_16), "ray-driven Siddon, 4x4 sub-rays per pixel (area reference)"),
                            (strip(s_1), "ray-driven Siddon, 1 ray per pixel"),
                            (strip(s_sf) - strip(s_16), "SF minus 16-subray reference")]):
    im = a.imshow(img, aspect="auto", cmap="gray" if "minus" not in t else "RdBu_r",
                  vmin=0 if "minus" not in t else -0.05 * vmax, vmax=vmax if "minus" not in t else 0.05 * vmax)
    a.set_title(f"{t}   (source 0, {mps} panels)", fontsize=10); a.set_ylabel("row"); fig.colorbar(im, ax=a, fraction=0.02)
ax[-1].set_xlabel("panel*n_u + column")
fig.suptitle("Sphere phantom sinograms, line integral (mm)", fontsize=13)
fig.savefig(os.path.join(out_dir, "voxel_projector_sphere_sinogram.png"), dpi=100); plt.close(fig)

# ------------------------------------------------------------- 8. uniform volume
P("\n[8] uniform volume (all ones): SF vs Siddon")
ones = torch.ones(n_x, n_y, n_z, device=dev)
u_sf = vproj.forward_project(ones); u_16 = ray_subray(ones); u_1 = rproj.forward_project(ones)
sup = u_16 > 0.05 * u_16.max()
P(f"  SF vs 16-subray: rel RMS {rel_rms(u_sf, u_16, sup):.4f};  SF vs 1-ray: rel RMS {rel_rms(u_sf, u_1, sup):.4f};  "
  f"max path SF {u_sf.max():.2f} vs Siddon {u_1.max():.2f} mm (volume is {n_x*vox:.0f} mm wide)")
ok8 = rel_rms(u_sf, u_16, sup) < 0.03
P(f"  {status(ok8)}")

# ------------------------------------------------------------------- 9. timing
P("\n[9] timing on a larger geometry")
T_n_x, T_n_y, T_n_z, T_vox = 64, 64, 16, 2.0
T_M, T_b = grid_affine(T_n_x, T_n_y, T_n_z, T_vox)
_, _, _, _, T_views, _ = build_geometry(32, 24, 6, 32, 32, 1.8, 2.0)
tv = VoxelProjector3D(T_n_x, T_n_y, T_n_z, T_M, T_b, T_views, device=dev)
P(f"  volume {T_n_x}x{T_n_y}x{T_n_z} ({tv.n_voxel:,} voxels), {tv.n_view} views, {tv.n_pix:,} pixels")
xv = torch.rand(T_n_x, T_n_y, T_n_z, device=dev); yv = torch.rand(tv.n_pix, device=dev)
def timeit(fn, n=5):
    fn(); sync(); t0 = time.time()
    for _ in range(n): fn()
    sync(); return (time.time() - t0) / n * 1e3
t_f_otf = timeit(lambda: tv.forward_project(xv)); t_b_otf = timeit(lambda: tv.back_project(yv))
sync(); t0 = time.time(); nnz_t = tv.precompute_footprints(); sync(); t_pre = (time.time() - t0) * 1e3
t_f_csr = timeit(lambda: tv.forward_project(xv)); t_b_csr = timeit(lambda: tv.back_project(yv))
ts, td = tv.rays(); sync(); t0 = time.time()
tr = make_ray_projector(T_n_x, T_n_y, T_n_z, T_M, T_b, ts, td, compressed=True); sync(); t_tv = (time.time() - t0) * 1e3
t_f_ray = timeit(lambda: tr.forward_project(xv)); t_b_ray = timeit(lambda: tr.back_project(yv))
P(f"  {'method':32s} {'precompute':>11s} {'forward':>9s} {'back':>9s}")
P(f"  {'voxel SF, on-the-fly':32s} {'-':>11s} {t_f_otf:8.1f}ms {t_b_otf:8.1f}ms")
P(f"  {'voxel SF, CSR footprints':32s} {t_pre:9.1f}ms {t_f_csr:8.1f}ms {t_b_csr:8.1f}ms   nnz {nnz_t:,} ({tv.footprint_bytes/1e6:.0f} MB)")
P(f"  {'ray Siddon, compressed tvals':32s} {t_tv:9.1f}ms {t_f_ray:8.1f}ms {t_b_ray:8.1f}ms   tvals {tr.tvals_uint16.numel()*2/1e6:.0f} MB")
lhs, rhs, e_t = adjoint_err(tv, xv, yv); P(f"  adjoint rel err at this size: {e_t:.2e}")

# ------------------------------------------------------------ 10. CGLS recon
P("\n[10] CGLS reconstruction of the sphere with the voxel projector (30 iterations)")
def cgls(A, AT, y, n_it):
    x = torch.zeros(n_x, n_y, n_z, device=dev); r = y.clone(); s = AT(r); p = s.clone(); g = (s * s).sum()
    for _ in range(n_it):
        q = A(p); a = g / (q * q).sum().clamp_min(1e-30); x = x + a * p; r = r - a * q
        s = AT(r); g_new = (s * s).sum(); p = s + (g_new / g) * p; g = g_new
    return x
y_meas = s_16                                             # "measured": area-integrating detector
x_sf = cgls(vproj.forward_project, vproj.back_project, y_meas, 30)
x_ray = cgls(rproj.forward_project, rproj.back_project, y_meas, 30)
sampled = torch.zeros_like(sph, dtype=torch.bool); sampled[:, :, 2:-2] = True     # inside the cone
zz = torch.arange(n_z, device=dev); xx = (torch.arange(n_x, device=dev) - (n_x - 1) / 2) * vox
XX, YY = torch.meshgrid(xx, xx, indexing="ij"); inside = (torch.sqrt(XX**2 + YY**2) < 50)[..., None] & sampled
e_sf = float((x_sf - sph)[inside].pow(2).mean().sqrt()); e_ray = float((x_ray - sph)[inside].pow(2).mean().sqrt())
P(f"  RMSE inside r<50mm, sampled slices:  voxel-SF recon {e_sf:.4f}   ray-Siddon recon {e_ray:.4f}   (phantom max 1.0)")
fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True); k = n_z // 2
for a, (img, t) in zip(ax, [(sph, "phantom"), (x_sf, f"CGLS, voxel SF  (RMSE {e_sf:.3f})"), (x_ray, f"CGLS, ray Siddon (RMSE {e_ray:.3f})")]):
    im = a.imshow(img[:, :, k].T.cpu().numpy(), cmap="gray", vmin=-0.1, vmax=1.1, origin="lower"); a.set_title(t); fig.colorbar(im, ax=a, fraction=0.046)
fig.suptitle(f"central slice z-index {k}; data = 16-subray Siddon sinogram", fontsize=12)
fig.savefig(os.path.join(out_dir, "voxel_projector_cgls_recon.png"), dpi=100); plt.close(fig)
ok10 = e_sf < 0.15

# ------------------------------------------------------------------ summary
P("\n" + "=" * 78)
results = [("hierarchy", ok1), ("normalisation", ok2), ("adjoint", ok3), ("CSR==on-the-fly", ok4),
           ("autograd", ok5), ("valid mask", ok6), ("sphere vs Siddon", ok7), ("uniform vs Siddon", ok8),
           ("CGLS recon", ok10)]
for name, ok in results: P(f"  {name:20s} {status(ok)}")
P(f"  overall: {status(all(ok for _, ok in results))}")
P("=" * 78)
with open(os.path.join(out_dir, "voxel_projector_3d_results.txt"), "w") as f: f.write("\n".join(LOG) + "\n")
print(f"\nwrote {out_dir}/voxel_projector_3d_results.txt, voxel_projector_sphere_sinogram.png, voxel_projector_cgls_recon.png")
sys.exit(0 if all(ok for _, ok in results) else 1)
