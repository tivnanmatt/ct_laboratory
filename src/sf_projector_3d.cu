// File: src/sf_projector_3d.cu
//
// Voxel-driven 3-D projector / back-projector using SEPARABLE FOOTPRINTS
// (SF-TR: trapezoid transaxial x rectangle axial; Long, Fessler & Balter,
//  "3D forward and back-projection for X-ray CT using separable footprints",
//  IEEE Trans. Med. Imaging 29(11):1839-1850, 2010).
//
// This is the voxel-driven counterpart of the ray-driven (Siddon / tvals)
// projector in ct_projector_3d.cu.  Every kernel here is VOXEL-parallel:
// one thread owns one (batch, voxel) pair and either
//   forward : scatters  volume[v] * w  into  sino[pixel]   (atomicAdd on sino)
//   back    : gathers   sino[pixel] * w into  volume[v]    (NO atomics at all)
//
// ---------------------------------------------------------------------------
// Detector model
// ---------------------------------------------------------------------------
// The detector is a list of flat "views".  A view = (source, flat panel):
//   src (3)   source position (world, mm)
//   cen (3)   panel centre
//   u   (3)   unit column direction   (transaxial, "s" in the SF paper)
//   v   (3)   unit row    direction   (axial,      "t" in the SF paper)
//   n   (3)   panel normal = u x v
//   du, dv    pixel pitch along u, v (mm)
//   n_u, n_v  pixels along u, v
// packed as VIEW_STRIDE = 19 floats per view.  Pixel (iu, iv) centre is
//   cen + (iu - (n_u-1)/2) du u + (iv - (n_v-1)/2) dv v
// and the flat sinogram index (matches ct_laboratory static-CT ray order:
// column outer, row inner) is
//   idx = view_off[view] + iu * n_v + iv
// where view_off is the exclusive prefix sum of n_u*n_v over views.
//
// ---------------------------------------------------------------------------
// Voxel model  (M, b row-major, world = M @ (i,j,k) + b, same as the ray code)
// ---------------------------------------------------------------------------
// A voxel is the parallelepiped spanned by the columns of M, centred at
// M(i,j,k)+b.  Axis k (third column of M) is treated as the AXIAL direction.
//   transaxial footprint F1(u): unit-plateau trapezoid whose breakpoints are the
//       sorted projected u of the 4 corners  c +- ex/2 +- ey/2  (at k-centre)
//   axial footprint      F2(v): unit rectangle between the projected v of the 2
//       k-faces  c +- ez/2  (at (i,j)-centre)
//   amplitude l          : chord length of the CENTRAL ray through the voxel
//       = min_a  Delta_a / |d . e_a|   over the three voxel axes a  (this is the
//       "A2" amplitude of the paper generalised to any cone angle / voxel shape)
//   pixel weight  w(iu,iv) = l * [int_{iu}^{iu+1} F1 du] * [int_{iv}^{iv+1} F2 dv]
// with u, v measured in CONTINUOUS PIXEL units (pixel iu covers [iu, iu+1)), so
// the bracketed integrals are the mean footprint over the pixel face and w has
// units of length (mm), exactly like a Siddon line integral.
//
// Two execution modes share the same footprint code, so they are bit-identical:
//   on-the-fly : footprints recomputed every call (no storage)
//   CSR        : footprints precomputed once into (ptr[int64], idx[int32], w[f32])
//                = "list of pixel indices and weights for each voxel"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <cstdint>

#define SF_THREADS 256
#define VIEW_STRIDE 19

// ---------------------------------------------------------------------------
// view record
// ---------------------------------------------------------------------------
struct SFView {
    float sx, sy, sz;     // source
    float cx, cy, cz;     // panel centre
    float ux, uy, uz;     // column unit vector
    float vx, vy, vz;     // row unit vector
    float nx, ny, nz;     // normal
    float du, dv;         // pitches
    int   nu, nv;         // sizes
};

__device__ __forceinline__ SFView sf_load_view(const float* views, int w)
{
    const float* p = views + (size_t)w * VIEW_STRIDE;
    SFView V;
    V.sx = p[0];  V.sy = p[1];  V.sz = p[2];
    V.cx = p[3];  V.cy = p[4];  V.cz = p[5];
    V.ux = p[6];  V.uy = p[7];  V.uz = p[8];
    V.vx = p[9];  V.vy = p[10]; V.vz = p[11];
    V.nx = p[12]; V.ny = p[13]; V.nz = p[14];
    V.du = p[15]; V.dv = p[16];
    V.nu = (int)p[17]; V.nv = (int)p[18];
    return V;
}

// Project world point P from the view's source onto the panel plane.
// Returns continuous pixel coordinates (up, vp): pixel iu covers [iu, iu+1).
// Also returns t = magnification factor (panel distance / point distance along the ray).
__device__ __forceinline__ bool sf_project_point(
    const SFView& V, float px, float py, float pz,
    float& up, float& vp, float& t)
{
    float rx = px - V.sx, ry = py - V.sy, rz = pz - V.sz;
    float denom = rx * V.nx + ry * V.ny + rz * V.nz;
    if (fabsf(denom) < 1e-12f) return false;
    t = ((V.cx - V.sx) * V.nx + (V.cy - V.sy) * V.ny + (V.cz - V.sz) * V.nz) / denom;
    if (t <= 0.f) return false;                       // panel behind the source or point beyond it
    float hx = V.sx + t * rx - V.cx;
    float hy = V.sy + t * ry - V.cy;
    float hz = V.sz + t * rz - V.cz;
    float u = hx * V.ux + hy * V.uy + hz * V.uz;
    float v = hx * V.vx + hy * V.vy + hz * V.vz;
    up = u / V.du + 0.5f * (float)(V.nu - 1) + 0.5f;
    vp = v / V.dv + 0.5f * (float)(V.nv - 1) + 0.5f;
    return true;
}

// Cumulative integral G(x) = int_{-inf}^{x} of the unit-plateau trapezoid
// with breakpoints t0 <= t1 <= t2 <= t3 (rise on [t0,t1], flat on [t1,t2], fall on [t2,t3]).
__device__ __forceinline__ float sf_trap_cdf(float x, float t0, float t1, float t2, float t3)
{
    if (x <= t0) return 0.f;
    const float A1 = 0.5f * (t1 - t0);
    const float A2 = (t2 - t1);
    const float A3 = 0.5f * (t3 - t2);
    if (x < t1) { float d = t1 - t0; float y = x - t0; return (d > 0.f) ? 0.5f * y * y / d : 0.f; }
    if (x < t2) { return A1 + (x - t1); }
    if (x < t3) { float d = t3 - t2; float y = x - t2; return A1 + A2 + ((d > 0.f) ? (y - 0.5f * y * y / d) : 0.f); }
    return A1 + A2 + A3;
}
__device__ __forceinline__ float sf_trap_int(float a, float b, float t0, float t1, float t2, float t3)
{
    return sf_trap_cdf(b, t0, t1, t2, t3) - sf_trap_cdf(a, t0, t1, t2, t3);
}
// Integral of the unit rectangle [r0, r1] over [a, b]
__device__ __forceinline__ float sf_rect_int(float a, float b, float r0, float r1)
{
    float lo = fmaxf(a, r0), hi = fminf(b, r1);
    return (hi > lo) ? (hi - lo) : 0.f;
}

// ---------------------------------------------------------------------------
// Separable footprint of one voxel in one view
// ---------------------------------------------------------------------------
struct SFFootprint {
    float t0, t1, t2, t3;   // trapezoid breakpoints (pixel units, u)
    float r0, r1;           // rectangle (pixel units, v)
    float amp;              // amplitude (mm)
    int   iu0, iu1, iv0, iv1;
    bool  ok;
};

// cx,cy,cz: voxel centre.  e?x,e?y,e?z: HALF-edge vectors (columns of M / 2).
// lx,ly,lz: full edge lengths.  hd: half body-diagonal (for the cheap reject).
__device__ __forceinline__ SFFootprint sf_footprint(
    const SFView& V,
    float cx, float cy, float cz,
    float exx, float exy, float exz,
    float eyx, float eyy, float eyz,
    float ezx, float ezy, float ezz,
    float lx, float ly, float lz, float hd)
{
    SFFootprint F; F.ok = false;

    // ---- cheap reject: project the centre, bound the footprint by the half-diagonal
    float uc, vc, t;
    if (!sf_project_point(V, cx, cy, cz, uc, vc, t)) return F;
    float mu = hd * t / V.du + 1.f, mv = hd * t / V.dv + 1.f;
    if (uc < -mu || uc > (float)V.nu + mu || vc < -mv || vc > (float)V.nv + mv) return F;

    // ---- amplitude: chord through the centre along the dominant voxel axis
    float dx = cx - V.sx, dy = cy - V.sy, dz = cz - V.sz;
    float dn = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dn < 1e-9f) return F;
    dx /= dn; dy /= dn; dz /= dn;
    float ax = fabsf((dx * exx + dy * exy + dz * exz) * 2.f / lx);   // |d . e_x_hat|
    float ay = fabsf((dx * eyx + dy * eyy + dz * eyz) * 2.f / ly);
    float az = fabsf((dx * ezx + dy * ezy + dz * ezz) * 2.f / lz);
    float amp = 1e30f;
    if (ax > 1e-9f) amp = fminf(amp, lx / ax);
    if (ay > 1e-9f) amp = fminf(amp, ly / ay);
    if (az > 1e-9f) amp = fminf(amp, lz / az);
    if (amp > 1e29f) return F;
    F.amp = amp;

    // ---- transaxial trapezoid: 4 corners at k-centre
    float us[4], vtmp, ttmp;
    const float s1[4] = { 1.f,  1.f, -1.f, -1.f };
    const float s2[4] = { 1.f, -1.f,  1.f, -1.f };
    for (int c = 0; c < 4; ++c) {
        float px = cx + s1[c] * exx + s2[c] * eyx;
        float py = cy + s1[c] * exy + s2[c] * eyy;
        float pz = cz + s1[c] * exz + s2[c] * eyz;
        if (!sf_project_point(V, px, py, pz, us[c], vtmp, ttmp)) return F;
    }
    // sorting network for 4
    float tmp;
    if (us[0] > us[1]) { tmp = us[0]; us[0] = us[1]; us[1] = tmp; }
    if (us[2] > us[3]) { tmp = us[2]; us[2] = us[3]; us[3] = tmp; }
    if (us[0] > us[2]) { tmp = us[0]; us[0] = us[2]; us[2] = tmp; }
    if (us[1] > us[3]) { tmp = us[1]; us[1] = us[3]; us[3] = tmp; }
    if (us[1] > us[2]) { tmp = us[1]; us[1] = us[2]; us[2] = tmp; }
    F.t0 = us[0]; F.t1 = us[1]; F.t2 = us[2]; F.t3 = us[3];

    // ---- axial rectangle: the 2 k-faces at (i,j)-centre
    float utmp, v0, v1;
    if (!sf_project_point(V, cx + ezx, cy + ezy, cz + ezz, utmp, v0, ttmp)) return F;
    if (!sf_project_point(V, cx - ezx, cy - ezy, cz - ezz, utmp, v1, ttmp)) return F;
    F.r0 = fminf(v0, v1); F.r1 = fmaxf(v0, v1);

    // ---- pixel index ranges (pixel iu overlaps [t0,t3) iff floor(t0) <= iu <= ceil(t3)-1)
    F.iu0 = max(0, (int)floorf(F.t0));
    F.iu1 = min(V.nu - 1, (int)ceilf(F.t3) - 1);
    F.iv0 = max(0, (int)floorf(F.r0));
    F.iv1 = min(V.nv - 1, (int)ceilf(F.r1) - 1);
    if (F.iu1 < F.iu0 || F.iv1 < F.iv0) return F;
    F.ok = true;
    return F;
}

// voxel geometry from (i,j,k) and M,b
__device__ __forceinline__ void sf_voxel_centre(
    const float* M, const float* b, int i, int j, int k,
    float& cx, float& cy, float& cz)
{
    cx = M[0] * i + M[1] * j + M[2] * k + b[0];
    cy = M[3] * i + M[4] * j + M[5] * k + b[1];
    cz = M[6] * i + M[7] * j + M[8] * k + b[2];
}

#define SF_VOXEL_EDGES(M)                                               \
    const float exx = 0.5f * M[0], exy = 0.5f * M[3], exz = 0.5f * M[6]; \
    const float eyx = 0.5f * M[1], eyy = 0.5f * M[4], eyz = 0.5f * M[7]; \
    const float ezx = 0.5f * M[2], ezy = 0.5f * M[5], ezz = 0.5f * M[8]; \
    const float lx = 2.f * sqrtf(exx * exx + exy * exy + exz * exz);    \
    const float ly = 2.f * sqrtf(eyx * eyx + eyy * eyy + eyz * eyz);    \
    const float lz = 2.f * sqrtf(ezx * ezx + ezy * ezy + ezz * ezz);    \
    const float hd = 0.5f * sqrtf(lx * lx + ly * ly + lz * lz);

// ===========================================================================
// ON-THE-FLY kernels
// ===========================================================================
__global__ void sf_forward_kernel(
    const float* __restrict__ vol, int batch, int nx, int ny, int nz,
    const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, int n_view, const int64_t* __restrict__ view_off,
    const uint8_t* __restrict__ valid, int has_valid,
    float* __restrict__ sino, int64_t n_pix)
{
    const int64_t n_vox = (int64_t)nx * ny * nz;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)batch * n_vox) return;
    const int bi = (int)(gid / n_vox);
    const int64_t v = gid % n_vox;
    const float val = vol[gid];
    if (val == 0.f) return;                                   // nothing to scatter
    const int i = (int)(v / ((int64_t)ny * nz));
    const int j = (int)((v / nz) % ny);
    const int k = (int)(v % nz);
    float cx, cy, cz; sf_voxel_centre(M, b, i, j, k, cx, cy, cz);
    SF_VOXEL_EDGES(M)
    float* sb = sino + (int64_t)bi * n_pix;

    for (int w = 0; w < n_view; ++w) {
        SFView V = sf_load_view(views, w);
        SFFootprint F = sf_footprint(V, cx, cy, cz, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, lx, ly, lz, hd);
        if (!F.ok) continue;
        const int64_t off = view_off[w];
        for (int iu = F.iu0; iu <= F.iu1; ++iu) {
            float wu = sf_trap_int((float)iu, (float)iu + 1.f, F.t0, F.t1, F.t2, F.t3);
            if (wu <= 0.f) continue;
            for (int iv = F.iv0; iv <= F.iv1; ++iv) {
                float wv = sf_rect_int((float)iv, (float)iv + 1.f, F.r0, F.r1);
                if (wv <= 0.f) continue;
                int64_t pidx = off + (int64_t)iu * V.nv + iv;
                if (has_valid && !valid[pidx]) continue;
                atomicAdd(&sb[pidx], val * F.amp * wu * wv);
            }
        }
    }
}

__global__ void sf_back_kernel(
    const float* __restrict__ sino, int batch, int nx, int ny, int nz,
    const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, int n_view, const int64_t* __restrict__ view_off,
    const uint8_t* __restrict__ valid, int has_valid,
    float* __restrict__ vol, int64_t n_pix)
{
    const int64_t n_vox = (int64_t)nx * ny * nz;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)batch * n_vox) return;
    const int bi = (int)(gid / n_vox);
    const int64_t v = gid % n_vox;
    const int i = (int)(v / ((int64_t)ny * nz));
    const int j = (int)((v / nz) % ny);
    const int k = (int)(v % nz);
    float cx, cy, cz; sf_voxel_centre(M, b, i, j, k, cx, cy, cz);
    SF_VOXEL_EDGES(M)
    const float* sb = sino + (int64_t)bi * n_pix;
    float acc = 0.f;

    for (int w = 0; w < n_view; ++w) {
        SFView V = sf_load_view(views, w);
        SFFootprint F = sf_footprint(V, cx, cy, cz, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, lx, ly, lz, hd);
        if (!F.ok) continue;
        const int64_t off = view_off[w];
        for (int iu = F.iu0; iu <= F.iu1; ++iu) {
            float wu = sf_trap_int((float)iu, (float)iu + 1.f, F.t0, F.t1, F.t2, F.t3);
            if (wu <= 0.f) continue;
            for (int iv = F.iv0; iv <= F.iv1; ++iv) {
                float wv = sf_rect_int((float)iv, (float)iv + 1.f, F.r0, F.r1);
                if (wv <= 0.f) continue;
                int64_t pidx = off + (int64_t)iu * V.nv + iv;
                if (has_valid && !valid[pidx]) continue;
                acc += sb[pidx] * F.amp * wu * wv;
            }
        }
    }
    vol[gid] = acc;                                           // no atomics: thread owns the voxel
}

// ===========================================================================
// CSR precompute: per-voxel list of (pixel index, weight)
// ===========================================================================
// pass 1: count entries per voxel
__global__ void sf_count_kernel(
    int nx, int ny, int nz,
    const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, int n_view, const int64_t* __restrict__ view_off,
    const uint8_t* __restrict__ valid, int has_valid,
    int64_t* __restrict__ counts)
{
    const int64_t n_vox = (int64_t)nx * ny * nz;
    const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;
    const int i = (int)(v / ((int64_t)ny * nz));
    const int j = (int)((v / nz) % ny);
    const int k = (int)(v % nz);
    float cx, cy, cz; sf_voxel_centre(M, b, i, j, k, cx, cy, cz);
    SF_VOXEL_EDGES(M)
    int64_t cnt = 0;
    for (int w = 0; w < n_view; ++w) {
        SFView V = sf_load_view(views, w);
        SFFootprint F = sf_footprint(V, cx, cy, cz, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, lx, ly, lz, hd);
        if (!F.ok) continue;
        const int64_t off = view_off[w];
        for (int iu = F.iu0; iu <= F.iu1; ++iu) {
            float wu = sf_trap_int((float)iu, (float)iu + 1.f, F.t0, F.t1, F.t2, F.t3);
            if (wu <= 0.f) continue;
            for (int iv = F.iv0; iv <= F.iv1; ++iv) {
                float wv = sf_rect_int((float)iv, (float)iv + 1.f, F.r0, F.r1);
                if (wv <= 0.f) continue;
                int64_t pidx = off + (int64_t)iu * V.nv + iv;
                if (has_valid && !valid[pidx]) continue;
                ++cnt;
            }
        }
    }
    counts[v] = cnt;
}

// pass 2: fill (identical enumeration order to pass 1)
__global__ void sf_fill_kernel(
    int nx, int ny, int nz,
    const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, int n_view, const int64_t* __restrict__ view_off,
    const uint8_t* __restrict__ valid, int has_valid,
    const int64_t* __restrict__ ptr, int32_t* __restrict__ idx, float* __restrict__ wgt)
{
    const int64_t n_vox = (int64_t)nx * ny * nz;
    const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n_vox) return;
    const int i = (int)(v / ((int64_t)ny * nz));
    const int j = (int)((v / nz) % ny);
    const int k = (int)(v % nz);
    float cx, cy, cz; sf_voxel_centre(M, b, i, j, k, cx, cy, cz);
    SF_VOXEL_EDGES(M)
    int64_t e = ptr[v];
    for (int w = 0; w < n_view; ++w) {
        SFView V = sf_load_view(views, w);
        SFFootprint F = sf_footprint(V, cx, cy, cz, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, lx, ly, lz, hd);
        if (!F.ok) continue;
        const int64_t off = view_off[w];
        for (int iu = F.iu0; iu <= F.iu1; ++iu) {
            float wu = sf_trap_int((float)iu, (float)iu + 1.f, F.t0, F.t1, F.t2, F.t3);
            if (wu <= 0.f) continue;
            for (int iv = F.iv0; iv <= F.iv1; ++iv) {
                float wv = sf_rect_int((float)iv, (float)iv + 1.f, F.r0, F.r1);
                if (wv <= 0.f) continue;
                int64_t pidx = off + (int64_t)iu * V.nv + iv;
                if (has_valid && !valid[pidx]) continue;
                idx[e] = (int32_t)pidx;
                wgt[e] = F.amp * wu * wv;
                ++e;
            }
        }
    }
}

// ===========================================================================
// CSR apply kernels
// ===========================================================================
__global__ void sf_forward_csr_kernel(
    const float* __restrict__ vol, int batch, int64_t n_vox,
    const int64_t* __restrict__ ptr, const int32_t* __restrict__ idx, const float* __restrict__ wgt,
    float* __restrict__ sino, int64_t n_pix)
{
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)batch * n_vox) return;
    const int bi = (int)(gid / n_vox);
    const int64_t v = gid % n_vox;
    const float val = vol[gid];
    if (val == 0.f) return;
    float* sb = sino + (int64_t)bi * n_pix;
    const int64_t e0 = ptr[v], e1 = ptr[v + 1];
    for (int64_t e = e0; e < e1; ++e) atomicAdd(&sb[idx[e]], val * wgt[e]);
}

__global__ void sf_back_csr_kernel(
    const float* __restrict__ sino, int batch, int64_t n_vox,
    const int64_t* __restrict__ ptr, const int32_t* __restrict__ idx, const float* __restrict__ wgt,
    float* __restrict__ vol, int64_t n_pix)
{
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)batch * n_vox) return;
    const int bi = (int)(gid / n_vox);
    const int64_t v = gid % n_vox;
    const float* sb = sino + (int64_t)bi * n_pix;
    const int64_t e0 = ptr[v], e1 = ptr[v + 1];
    float acc = 0.f;
    for (int64_t e = e0; e < e1; ++e) acc += sb[idx[e]] * wgt[e];
    vol[gid] = acc;
}

// ===========================================================================
// COLUMN-CACHED mode: per transaxial column (i,j), a culled list of the views
// whose footprint touches the column, with the TRANSAXIAL trapezoid stored.
// ===========================================================================
// Rationale.  For the ring, a voxel is touched by a few percent of the (source,
// module) views, so the on-the-fly kernels spend almost all their time rejecting
// views.  The trapezoid F1(u) of voxel (i,j,k) depends on k only through the
// (small) tilt of the panel rows away from the axial direction, and that
// dependence is projective in z, i.e. linear to second order over the volume
// height.  So the trapezoid is stored at the two END slices of the column and
// linearly interpolated in k per voxel; the axial rectangle F2(v) and the
// amplitude are still evaluated exactly per voxel.  Per entry:
//   col_view[e]      int32     view index
//   col_trap[2e,2e+1] float4x2 t0..t3 at k=0 and at k=nz-1 (pixel units)
// col_ptr[nx*ny+1] int64 is the CSR pointer over columns (36 B per entry).
// The view list is exact (a superset test); with axial panel rows the result
// is identical to the on-the-fly kernels, with tilted rows it agrees to O(tilt^2).

// transaxial trapezoid of the voxel centred at (cx,cy,cz)
__device__ __forceinline__ bool sf_trap_at(
    const SFView& V, float cx, float cy, float cz,
    float exx, float exy, float exz, float eyx, float eyy, float eyz, float4& trap)
{
    float us[4], vtmp, ttmp;
    const float s1[4] = { 1.f,  1.f, -1.f, -1.f };
    const float s2[4] = { 1.f, -1.f,  1.f, -1.f };
    for (int c = 0; c < 4; ++c) {
        if (!sf_project_point(V, cx + s1[c] * exx + s2[c] * eyx, cy + s1[c] * exy + s2[c] * eyy,
                              cz + s1[c] * exz + s2[c] * eyz, us[c], vtmp, ttmp)) return false;
    }
    float tmp;
    if (us[0] > us[1]) { tmp = us[0]; us[0] = us[1]; us[1] = tmp; }
    if (us[2] > us[3]) { tmp = us[2]; us[2] = us[3]; us[3] = tmp; }
    if (us[0] > us[2]) { tmp = us[0]; us[0] = us[2]; us[2] = tmp; }
    if (us[1] > us[3]) { tmp = us[1]; us[1] = us[3]; us[3] = tmp; }
    if (us[1] > us[2]) { tmp = us[1]; us[1] = us[2]; us[2] = tmp; }
    trap = make_float4(us[0], us[1], us[2], us[3]);
    return true;
}

// culling test for a whole column, exact for the stored (end-slice interpolated) model:
// cheap centre reject at the mid-slice with the column half-height as margin, then the
// u-range spanned by the two end-slice trapezoids must overlap the panel.  Also returns them.
__device__ __forceinline__ bool sf_column_test(
    const SFView& V, int nz, float cx, float cy, float cz,
    float c0x, float c0y, float c0z, float c1x, float c1y, float c1z,
    float exx, float exy, float exz, float eyx, float eyy, float eyz,
    float ezx, float ezy, float ezz, float hd, float4& tr0, float4& tr1)
{
    float uc, vc, t;
    if (!sf_project_point(V, cx, cy, cz, uc, vc, t)) return false;
    const float hz = (float)nz * sqrtf(ezx * ezx + ezy * ezy + ezz * ezz);   // column half-height
    const float mu = (hz + hd) * t / V.du + 1.f, mv = (hz + hd) * t / V.dv + 1.f;
    if (uc < -mu || uc > (float)V.nu + mu || vc < -mv || vc > (float)V.nv + mv) return false;
    if (!sf_trap_at(V, c0x, c0y, c0z, exx, exy, exz, eyx, eyy, eyz, tr0)) return false;
    if (!sf_trap_at(V, c1x, c1y, c1z, exx, exy, exz, eyx, eyy, eyz, tr1)) return false;
    const float ulo = fminf(tr0.x, tr1.x), uhi = fmaxf(tr0.w, tr1.w);
    return (ulo < (float)V.nu) && (uhi > 0.f);
}

// column centre at the mid-slice
__device__ __forceinline__ void sf_column_centre(
    const float* M, const float* b, int i, int j, int nz, float& cx, float& cy, float& cz)
{
    const float km = 0.5f * (float)(nz - 1);
    cx = M[0] * i + M[1] * j + M[2] * km + b[0];
    cy = M[3] * i + M[4] * j + M[5] * km + b[1];
    cz = M[6] * i + M[7] * j + M[8] * km + b[2];
}

__global__ void sf_col_count_kernel(
    int nx, int ny, int nz, const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, int n_view, int64_t* __restrict__ counts)
{
    const int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (int64_t)nx * ny) return;
    const int i = (int)(c / ny), j = (int)(c % ny);
    float cx, cy, cz; sf_column_centre(M, b, i, j, nz, cx, cy, cz);
    float c0x, c0y, c0z, c1x, c1y, c1z;
    sf_voxel_centre(M, b, i, j, 0, c0x, c0y, c0z);
    sf_voxel_centre(M, b, i, j, nz - 1, c1x, c1y, c1z);
    SF_VOXEL_EDGES(M)
    int64_t cnt = 0; float4 tr0, tr1;
    for (int w = 0; w < n_view; ++w) {
        SFView V = sf_load_view(views, w);
        if (sf_column_test(V, nz, cx, cy, cz, c0x, c0y, c0z, c1x, c1y, c1z, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, hd, tr0, tr1)) ++cnt;
    }
    counts[c] = cnt;
}

__global__ void sf_col_fill_kernel(
    int nx, int ny, int nz, const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, int n_view, const int64_t* __restrict__ ptr,
    int32_t* __restrict__ col_view, float4* __restrict__ col_trap)
{
    const int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= (int64_t)nx * ny) return;
    const int i = (int)(c / ny), j = (int)(c % ny);
    float cx, cy, cz; sf_column_centre(M, b, i, j, nz, cx, cy, cz);
    float c0x, c0y, c0z, c1x, c1y, c1z;
    sf_voxel_centre(M, b, i, j, 0, c0x, c0y, c0z);
    sf_voxel_centre(M, b, i, j, nz - 1, c1x, c1y, c1z);
    SF_VOXEL_EDGES(M)
    int64_t e = ptr[c]; float4 tr0, tr1;
    for (int w = 0; w < n_view; ++w) {
        SFView V = sf_load_view(views, w);
        if (!sf_column_test(V, nz, cx, cy, cz, c0x, c0y, c0z, c1x, c1y, c1z, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, hd, tr0, tr1)) continue;
        col_view[e] = w; col_trap[2 * e] = tr0; col_trap[2 * e + 1] = tr1; ++e;
    }
}

// per-voxel part of the footprint given the stored trapezoid: amplitude + axial rectangle
__device__ __forceinline__ bool sf_col_voxel(
    const SFView& V, const float4& tra, const float4& trb, float fk,
    float cx, float cy, float cz,
    float exx, float exy, float exz, float eyx, float eyy, float eyz, float ezx, float ezy, float ezz,
    float lx, float ly, float lz, SFFootprint& F)
{
    float dx = cx - V.sx, dy = cy - V.sy, dz = cz - V.sz;
    float dn = sqrtf(dx * dx + dy * dy + dz * dz);
    if (dn < 1e-9f) return false;
    dx /= dn; dy /= dn; dz /= dn;
    float ax = fabsf((dx * exx + dy * exy + dz * exz) * 2.f / lx);
    float ay = fabsf((dx * eyx + dy * eyy + dz * eyz) * 2.f / ly);
    float az = fabsf((dx * ezx + dy * ezy + dz * ezz) * 2.f / lz);
    float amp = 1e30f;
    if (ax > 1e-9f) amp = fminf(amp, lx / ax);
    if (ay > 1e-9f) amp = fminf(amp, ly / ay);
    if (az > 1e-9f) amp = fminf(amp, lz / az);
    if (amp > 1e29f) return false;
    float utmp, v0, v1, ttmp;
    if (!sf_project_point(V, cx + ezx, cy + ezy, cz + ezz, utmp, v0, ttmp)) return false;
    if (!sf_project_point(V, cx - ezx, cy - ezy, cz - ezz, utmp, v1, ttmp)) return false;
    F.amp = amp;
    F.t0 = tra.x + fk * (trb.x - tra.x); F.t1 = tra.y + fk * (trb.y - tra.y);
    F.t2 = tra.z + fk * (trb.z - tra.z); F.t3 = tra.w + fk * (trb.w - tra.w);
    F.r0 = fminf(v0, v1); F.r1 = fmaxf(v0, v1);
    F.iu0 = max(0, (int)floorf(F.t0)); F.iu1 = min(V.nu - 1, (int)ceilf(F.t3) - 1);
    F.iv0 = max(0, (int)floorf(F.r0)); F.iv1 = min(V.nv - 1, (int)ceilf(F.r1) - 1);
    return (F.iu1 >= F.iu0) && (F.iv1 >= F.iv0);
}

__global__ void sf_forward_col_kernel(
    const float* __restrict__ vol, int batch, int nx, int ny, int nz,
    const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, const int64_t* __restrict__ view_off,
    const uint8_t* __restrict__ valid, int has_valid,
    const int64_t* __restrict__ col_ptr, const int32_t* __restrict__ col_view, const float4* __restrict__ col_trap,
    float* __restrict__ sino, int64_t n_pix)
{
    const int64_t n_vox = (int64_t)nx * ny * nz;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)batch * n_vox) return;
    const int bi = (int)(gid / n_vox);
    const int64_t v = gid % n_vox;
    const float val = vol[gid];
    if (val == 0.f) return;
    const int64_t c = v / nz; const int k = (int)(v % nz);
    const int i = (int)(c / ny), j = (int)(c % ny);
    float cx, cy, cz; sf_voxel_centre(M, b, i, j, k, cx, cy, cz);
    SF_VOXEL_EDGES(M)
    const float fk = (nz > 1) ? (float)k / (float)(nz - 1) : 0.f;
    float* sb = sino + (int64_t)bi * n_pix;
    const int64_t e0 = col_ptr[c], e1 = col_ptr[c + 1];
    for (int64_t e = e0; e < e1; ++e) {
        const int w = col_view[e];
        SFView V = sf_load_view(views, w);
        SFFootprint F;
        if (!sf_col_voxel(V, col_trap[2 * e], col_trap[2 * e + 1], fk, cx, cy, cz, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, lx, ly, lz, F)) continue;
        const int64_t off = view_off[w];
        for (int iu = F.iu0; iu <= F.iu1; ++iu) {
            float wu = sf_trap_int((float)iu, (float)iu + 1.f, F.t0, F.t1, F.t2, F.t3);
            if (wu <= 0.f) continue;
            for (int iv = F.iv0; iv <= F.iv1; ++iv) {
                float wv = sf_rect_int((float)iv, (float)iv + 1.f, F.r0, F.r1);
                if (wv <= 0.f) continue;
                int64_t pidx = off + (int64_t)iu * V.nv + iv;
                if (has_valid && !valid[pidx]) continue;
                atomicAdd(&sb[pidx], val * F.amp * wu * wv);
            }
        }
    }
}

__global__ void sf_back_col_kernel(
    const float* __restrict__ sino, int batch, int nx, int ny, int nz,
    const float* __restrict__ M, const float* __restrict__ b,
    const float* __restrict__ views, const int64_t* __restrict__ view_off,
    const uint8_t* __restrict__ valid, int has_valid,
    const int64_t* __restrict__ col_ptr, const int32_t* __restrict__ col_view, const float4* __restrict__ col_trap,
    float* __restrict__ vol, int64_t n_pix)
{
    const int64_t n_vox = (int64_t)nx * ny * nz;
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)batch * n_vox) return;
    const int bi = (int)(gid / n_vox);
    const int64_t v = gid % n_vox;
    const int64_t c = v / nz; const int k = (int)(v % nz);
    const int i = (int)(c / ny), j = (int)(c % ny);
    float cx, cy, cz; sf_voxel_centre(M, b, i, j, k, cx, cy, cz);
    SF_VOXEL_EDGES(M)
    const float fk = (nz > 1) ? (float)k / (float)(nz - 1) : 0.f;
    const float* sb = sino + (int64_t)bi * n_pix;
    const int64_t e0 = col_ptr[c], e1 = col_ptr[c + 1];
    float acc = 0.f;
    for (int64_t e = e0; e < e1; ++e) {
        const int w = col_view[e];
        SFView V = sf_load_view(views, w);
        SFFootprint F;
        if (!sf_col_voxel(V, col_trap[2 * e], col_trap[2 * e + 1], fk, cx, cy, cz, exx, exy, exz, eyx, eyy, eyz, ezx, ezy, ezz, lx, ly, lz, F)) continue;
        const int64_t off = view_off[w];
        for (int iu = F.iu0; iu <= F.iu1; ++iu) {
            float wu = sf_trap_int((float)iu, (float)iu + 1.f, F.t0, F.t1, F.t2, F.t3);
            if (wu <= 0.f) continue;
            for (int iv = F.iv0; iv <= F.iv1; ++iv) {
                float wv = sf_rect_int((float)iv, (float)iv + 1.f, F.r0, F.r1);
                if (wv <= 0.f) continue;
                int64_t pidx = off + (int64_t)iu * V.nv + iv;
                if (has_valid && !valid[pidx]) continue;
                acc += sb[pidx] * F.amp * wu * wv;
            }
        }
    }
    vol[gid] = acc;
}

// ===========================================================================
// Host wrappers
// ===========================================================================
static inline void sf_check_common(const torch::Tensor& M, const torch::Tensor& b,
                                   const torch::Tensor& views, const torch::Tensor& view_off,
                                   const torch::Tensor& valid)
{
    TORCH_CHECK(M.is_cuda() && b.is_cuda() && views.is_cuda() && view_off.is_cuda(), "M,b,views,view_off must be CUDA");
    TORCH_CHECK(M.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32 && views.dtype() == torch::kFloat32, "M,b,views must be float32");
    TORCH_CHECK(view_off.dtype() == torch::kInt64, "view_off must be int64");
    TORCH_CHECK(M.numel() == 9 && b.numel() == 3, "M must be 3x3, b must be 3");
    TORCH_CHECK(views.dim() == 2 && views.size(1) == VIEW_STRIDE, "views must be [n_view, 19]");
    TORCH_CHECK(view_off.numel() == views.size(0) + 1, "view_off must be [n_view+1]");
    TORCH_CHECK(M.is_contiguous() && b.is_contiguous() && views.is_contiguous() && view_off.is_contiguous(), "inputs must be contiguous");
    if (valid.numel() > 0) {
        TORCH_CHECK(valid.is_cuda() && valid.dtype() == torch::kUInt8 && valid.is_contiguous(), "valid must be CUDA uint8 contiguous");
    }
}

static inline int64_t sf_n_pix(const torch::Tensor& view_off)
{
    return view_off[view_off.numel() - 1].item<int64_t>();
}

torch::Tensor sf_forward_project_3d_cuda(
    torch::Tensor volume, torch::Tensor M, torch::Tensor b,
    torch::Tensor views, torch::Tensor view_off, torch::Tensor valid)
{
    TORCH_CHECK(volume.is_cuda() && volume.dtype() == torch::kFloat32 && volume.is_contiguous(), "volume must be CUDA float32 contiguous");
    sf_check_common(M, b, views, view_off, valid);
    int64_t batch = 1, nx, ny, nz;
    if (volume.dim() == 3) { nx = volume.size(0); ny = volume.size(1); nz = volume.size(2); }
    else if (volume.dim() == 4) { batch = volume.size(0); nx = volume.size(1); ny = volume.size(2); nz = volume.size(3); }
    else TORCH_CHECK(false, "volume must be [X,Y,Z] or [B,X,Y,Z]");
    const int64_t n_pix = sf_n_pix(view_off);
    const int n_view = (int)views.size(0);
    auto sino = torch::zeros({batch, n_pix}, volume.options());
    const int64_t total = batch * nx * ny * nz;
    const int blocks = (int)((total + SF_THREADS - 1) / SF_THREADS);
    const int has_valid = valid.numel() > 0 ? 1 : 0;
    sf_forward_kernel<<<blocks, SF_THREADS>>>(
        volume.data_ptr<float>(), (int)batch, (int)nx, (int)ny, (int)nz,
        M.data_ptr<float>(), b.data_ptr<float>(),
        views.data_ptr<float>(), n_view, view_off.data_ptr<int64_t>(),
        has_valid ? valid.data_ptr<uint8_t>() : nullptr, has_valid,
        sino.data_ptr<float>(), n_pix);
    return sino;
}

torch::Tensor sf_back_project_3d_cuda(
    torch::Tensor sino, torch::Tensor M, torch::Tensor b,
    torch::Tensor views, torch::Tensor view_off, torch::Tensor valid,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(sino.is_cuda() && sino.dtype() == torch::kFloat32 && sino.is_contiguous(), "sino must be CUDA float32 contiguous");
    sf_check_common(M, b, views, view_off, valid);
    int64_t batch = 1, n_pix_in;
    if (sino.dim() == 1) { n_pix_in = sino.size(0); }
    else if (sino.dim() == 2) { batch = sino.size(0); n_pix_in = sino.size(1); }
    else TORCH_CHECK(false, "sino must be [n_pix] or [B, n_pix]");
    const int64_t n_pix = sf_n_pix(view_off);
    TORCH_CHECK(n_pix_in == n_pix, "sino length does not match the views");
    const int n_view = (int)views.size(0);
    auto vol = torch::zeros({batch, nx, ny, nz}, sino.options());
    const int64_t total = batch * nx * ny * nz;
    const int blocks = (int)((total + SF_THREADS - 1) / SF_THREADS);
    const int has_valid = valid.numel() > 0 ? 1 : 0;
    sf_back_kernel<<<blocks, SF_THREADS>>>(
        sino.data_ptr<float>(), (int)batch, (int)nx, (int)ny, (int)nz,
        M.data_ptr<float>(), b.data_ptr<float>(),
        views.data_ptr<float>(), n_view, view_off.data_ptr<int64_t>(),
        has_valid ? valid.data_ptr<uint8_t>() : nullptr, has_valid,
        vol.data_ptr<float>(), n_pix);
    return vol;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sf_precompute_footprints_3d_cuda(
    int64_t nx, int64_t ny, int64_t nz,
    torch::Tensor M, torch::Tensor b,
    torch::Tensor views, torch::Tensor view_off, torch::Tensor valid)
{
    sf_check_common(M, b, views, view_off, valid);
    const int n_view = (int)views.size(0);
    const int64_t n_vox = nx * ny * nz;
    const int blocks = (int)((n_vox + SF_THREADS - 1) / SF_THREADS);
    const int has_valid = valid.numel() > 0 ? 1 : 0;
    auto counts = torch::zeros({n_vox}, torch::TensorOptions().dtype(torch::kInt64).device(M.device()));
    sf_count_kernel<<<blocks, SF_THREADS>>>(
        (int)nx, (int)ny, (int)nz, M.data_ptr<float>(), b.data_ptr<float>(),
        views.data_ptr<float>(), n_view, view_off.data_ptr<int64_t>(),
        has_valid ? valid.data_ptr<uint8_t>() : nullptr, has_valid,
        counts.data_ptr<int64_t>());
    auto ptr = torch::zeros({n_vox + 1}, counts.options());
    ptr.slice(0, 1, n_vox + 1) = torch::cumsum(counts, 0);
    const int64_t nnz = ptr[n_vox].item<int64_t>();
    auto idx = torch::empty({nnz}, torch::TensorOptions().dtype(torch::kInt32).device(M.device()));
    auto wgt = torch::empty({nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(M.device()));
    if (nnz > 0) {
        sf_fill_kernel<<<blocks, SF_THREADS>>>(
            (int)nx, (int)ny, (int)nz, M.data_ptr<float>(), b.data_ptr<float>(),
            views.data_ptr<float>(), n_view, view_off.data_ptr<int64_t>(),
            has_valid ? valid.data_ptr<uint8_t>() : nullptr, has_valid,
            ptr.data_ptr<int64_t>(), idx.data_ptr<int32_t>(), wgt.data_ptr<float>());
    }
    return std::make_tuple(ptr, idx, wgt);
}

torch::Tensor sf_forward_project_3d_csr_cuda(
    torch::Tensor volume, torch::Tensor ptr, torch::Tensor idx, torch::Tensor wgt, int64_t n_pix)
{
    TORCH_CHECK(volume.is_cuda() && volume.dtype() == torch::kFloat32 && volume.is_contiguous(), "volume must be CUDA float32 contiguous");
    TORCH_CHECK(ptr.is_cuda() && ptr.dtype() == torch::kInt64 && idx.is_cuda() && idx.dtype() == torch::kInt32 && wgt.is_cuda() && wgt.dtype() == torch::kFloat32, "bad CSR tensors");
    int64_t batch = 1, n_vox;
    if (volume.dim() == 3) { n_vox = volume.numel(); }
    else if (volume.dim() == 4) { batch = volume.size(0); n_vox = volume.numel() / batch; }
    else TORCH_CHECK(false, "volume must be [X,Y,Z] or [B,X,Y,Z]");
    TORCH_CHECK(ptr.numel() == n_vox + 1, "ptr length does not match the volume");
    auto sino = torch::zeros({batch, n_pix}, volume.options());
    const int64_t total = batch * n_vox;
    const int blocks = (int)((total + SF_THREADS - 1) / SF_THREADS);
    sf_forward_csr_kernel<<<blocks, SF_THREADS>>>(
        volume.data_ptr<float>(), (int)batch, n_vox,
        ptr.data_ptr<int64_t>(), idx.data_ptr<int32_t>(), wgt.data_ptr<float>(),
        sino.data_ptr<float>(), n_pix);
    return sino;
}

torch::Tensor sf_back_project_3d_csr_cuda(
    torch::Tensor sino, torch::Tensor ptr, torch::Tensor idx, torch::Tensor wgt,
    int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(sino.is_cuda() && sino.dtype() == torch::kFloat32 && sino.is_contiguous(), "sino must be CUDA float32 contiguous");
    TORCH_CHECK(ptr.is_cuda() && ptr.dtype() == torch::kInt64 && idx.is_cuda() && idx.dtype() == torch::kInt32 && wgt.is_cuda() && wgt.dtype() == torch::kFloat32, "bad CSR tensors");
    int64_t batch = 1, n_pix;
    if (sino.dim() == 1) { n_pix = sino.size(0); }
    else if (sino.dim() == 2) { batch = sino.size(0); n_pix = sino.size(1); }
    else TORCH_CHECK(false, "sino must be [n_pix] or [B, n_pix]");
    const int64_t n_vox = nx * ny * nz;
    TORCH_CHECK(ptr.numel() == n_vox + 1, "ptr length does not match the volume");
    auto vol = torch::zeros({batch, nx, ny, nz}, sino.options());
    const int64_t total = batch * n_vox;
    const int blocks = (int)((total + SF_THREADS - 1) / SF_THREADS);
    sf_back_csr_kernel<<<blocks, SF_THREADS>>>(
        sino.data_ptr<float>(), (int)batch, n_vox,
        ptr.data_ptr<int64_t>(), idx.data_ptr<int32_t>(), wgt.data_ptr<float>(),
        vol.data_ptr<float>(), n_pix);
    return vol;
}

// ---------------------------------------------------------------------------
// column-cache host wrappers
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sf_precompute_columns_3d_cuda(
    int64_t nx, int64_t ny, int64_t nz,
    torch::Tensor M, torch::Tensor b, torch::Tensor views, torch::Tensor view_off, torch::Tensor valid)
{
    sf_check_common(M, b, views, view_off, valid);
    const int n_view = (int)views.size(0);
    const int64_t n_col = nx * ny;
    const int blocks = (int)((n_col + SF_THREADS - 1) / SF_THREADS);
    auto counts = torch::zeros({n_col}, torch::TensorOptions().dtype(torch::kInt64).device(M.device()));
    sf_col_count_kernel<<<blocks, SF_THREADS>>>((int)nx, (int)ny, (int)nz, M.data_ptr<float>(), b.data_ptr<float>(),
                                                views.data_ptr<float>(), n_view, counts.data_ptr<int64_t>());
    auto ptr = torch::zeros({n_col + 1}, counts.options());
    ptr.slice(0, 1, n_col + 1) = torch::cumsum(counts, 0);
    const int64_t nnz = ptr[n_col].item<int64_t>();
    auto cv = torch::empty({nnz}, torch::TensorOptions().dtype(torch::kInt32).device(M.device()));
    auto ct = torch::empty({nnz, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(M.device()));
    if (nnz > 0) {
        sf_col_fill_kernel<<<blocks, SF_THREADS>>>((int)nx, (int)ny, (int)nz, M.data_ptr<float>(), b.data_ptr<float>(),
                                                   views.data_ptr<float>(), n_view, ptr.data_ptr<int64_t>(),
                                                   cv.data_ptr<int32_t>(), reinterpret_cast<float4*>(ct.data_ptr<float>()));
    }
    return std::make_tuple(ptr, cv, ct);
}

static inline void sf_check_cols(const torch::Tensor& col_ptr, const torch::Tensor& col_view, const torch::Tensor& col_trap, int64_t n_col)
{
    TORCH_CHECK(col_ptr.is_cuda() && col_ptr.dtype() == torch::kInt64 && col_ptr.is_contiguous() && col_ptr.numel() == n_col + 1, "bad col_ptr");
    TORCH_CHECK(col_view.is_cuda() && col_view.dtype() == torch::kInt32 && col_view.is_contiguous(), "bad col_view");
    TORCH_CHECK(col_trap.is_cuda() && col_trap.dtype() == torch::kFloat32 && col_trap.is_contiguous() && col_trap.dim() == 2 && col_trap.size(1) == 8 && col_trap.size(0) == col_view.numel(), "bad col_trap");
}

torch::Tensor sf_forward_project_3d_col_cuda(
    torch::Tensor volume, torch::Tensor M, torch::Tensor b, torch::Tensor views, torch::Tensor view_off, torch::Tensor valid,
    torch::Tensor col_ptr, torch::Tensor col_view, torch::Tensor col_trap)
{
    TORCH_CHECK(volume.is_cuda() && volume.dtype() == torch::kFloat32 && volume.is_contiguous(), "volume must be CUDA float32 contiguous");
    sf_check_common(M, b, views, view_off, valid);
    int64_t batch = 1, nx, ny, nz;
    if (volume.dim() == 3) { nx = volume.size(0); ny = volume.size(1); nz = volume.size(2); }
    else if (volume.dim() == 4) { batch = volume.size(0); nx = volume.size(1); ny = volume.size(2); nz = volume.size(3); }
    else TORCH_CHECK(false, "volume must be [X,Y,Z] or [B,X,Y,Z]");
    sf_check_cols(col_ptr, col_view, col_trap, nx * ny);
    const int64_t n_pix = sf_n_pix(view_off);
    auto sino = torch::zeros({batch, n_pix}, volume.options());
    const int64_t total = batch * nx * ny * nz;
    const int blocks = (int)((total + SF_THREADS - 1) / SF_THREADS);
    const int has_valid = valid.numel() > 0 ? 1 : 0;
    sf_forward_col_kernel<<<blocks, SF_THREADS>>>(
        volume.data_ptr<float>(), (int)batch, (int)nx, (int)ny, (int)nz, M.data_ptr<float>(), b.data_ptr<float>(),
        views.data_ptr<float>(), view_off.data_ptr<int64_t>(), has_valid ? valid.data_ptr<uint8_t>() : nullptr, has_valid,
        col_ptr.data_ptr<int64_t>(), col_view.data_ptr<int32_t>(), reinterpret_cast<const float4*>(col_trap.data_ptr<float>()),
        sino.data_ptr<float>(), n_pix);
    return sino;
}

torch::Tensor sf_back_project_3d_col_cuda(
    torch::Tensor sino, torch::Tensor M, torch::Tensor b, torch::Tensor views, torch::Tensor view_off, torch::Tensor valid,
    torch::Tensor col_ptr, torch::Tensor col_view, torch::Tensor col_trap, int64_t nx, int64_t ny, int64_t nz)
{
    TORCH_CHECK(sino.is_cuda() && sino.dtype() == torch::kFloat32 && sino.is_contiguous(), "sino must be CUDA float32 contiguous");
    sf_check_common(M, b, views, view_off, valid);
    sf_check_cols(col_ptr, col_view, col_trap, nx * ny);
    int64_t batch = 1, n_pix_in;
    if (sino.dim() == 1) { n_pix_in = sino.size(0); }
    else if (sino.dim() == 2) { batch = sino.size(0); n_pix_in = sino.size(1); }
    else TORCH_CHECK(false, "sino must be [n_pix] or [B, n_pix]");
    const int64_t n_pix = sf_n_pix(view_off);
    TORCH_CHECK(n_pix_in == n_pix, "sino length does not match the views");
    auto vol = torch::zeros({batch, nx, ny, nz}, sino.options());
    const int64_t total = batch * nx * ny * nz;
    const int blocks = (int)((total + SF_THREADS - 1) / SF_THREADS);
    const int has_valid = valid.numel() > 0 ? 1 : 0;
    sf_back_col_kernel<<<blocks, SF_THREADS>>>(
        sino.data_ptr<float>(), (int)batch, (int)nx, (int)ny, (int)nz, M.data_ptr<float>(), b.data_ptr<float>(),
        views.data_ptr<float>(), view_off.data_ptr<int64_t>(), has_valid ? valid.data_ptr<uint8_t>() : nullptr, has_valid,
        col_ptr.data_ptr<int64_t>(), col_view.data_ptr<int32_t>(), reinterpret_cast<const float4*>(col_trap.data_ptr<float>()),
        vol.data_ptr<float>(), n_pix);
    return vol;
}
