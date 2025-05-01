// File: src/ct_projector_3d.cu
//
// 2025-04-30  – changes:
//   • all kernels now expect **M_inv** (row-major, 9 floats)
//   • apply_affine_inverse_3d no longer inverts – it just multiplies by M_inv
//   • midpoint-to-voxel index uses IEEE-754 round-to-nearest-even
//   • host wrappers renamed arguments accordingly (but keep same function names
//     so the Python extension API remains unchanged)

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>

// -----------------------------------------------
// HELPER: apply_affine_inverse_3d
//   (x,y,z) → (i,j,k)   using pre-computed M_inv
// -----------------------------------------------
__device__ __forceinline__
void apply_affine_inverse_3d(
    float x, float y, float z,
    const float* M_inv,   // [9] = a11 … a33  (row-major)
    const float* b,       // [3]
    float& i_out, float& j_out, float& k_out)
{
    float x_shift = x - b[0];
    float y_shift = y - b[1];
    float z_shift = z - b[2];

    // layout: [0 1 2 ; 3 4 5 ; 6 7 8]
    i_out = M_inv[0]*x_shift + M_inv[1]*y_shift + M_inv[2]*z_shift;
    j_out = M_inv[3]*x_shift + M_inv[4]*y_shift + M_inv[5]*z_shift;
    k_out = M_inv[6]*x_shift + M_inv[7]*y_shift + M_inv[8]*z_shift;
}

// (the three kernels are unchanged except that the argument name is M_inv
//  and the rounding now uses __float2int_rn.)

// ……………………… compute_intersections_3d_kernel ………………………
__global__ void compute_intersections_3d_kernel(
    int n_x, int n_y, int n_z,
    const float* src_xyz, const float* dst_xyz, int n_ray,
    const float* M_inv, const float* b,
    float* t_out)
{
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= n_ray) return;

    float sx = src_xyz[3*ray_id+0], sy = src_xyz[3*ray_id+1], sz = src_xyz[3*ray_id+2];
    float dx = dst_xyz[3*ray_id+0], dy = dst_xyz[3*ray_id+1], dz = dst_xyz[3*ray_id+2];

    float si, sj, sk, di, dj, dk;
    apply_affine_inverse_3d(sx,sy,sz, M_inv,b, si,sj,sk);
    apply_affine_inverse_3d(dx,dy,dz, M_inv,b, di,dj,dk);

    float dirx = di-si, diry = dj-sj, dirz = dk-sk;
    int n_int = (n_x+1)+(n_y+1)+(n_z+1);
    float* t_vals = &t_out[ray_id*n_int];

    int cnt = 0;
    // x-planes
    for (int i=0;i<=n_x;++i){
        float plane = float(i)-0.5f;
        if (fabsf(dirx)<1e-12f) t_vals[cnt++] = INFINITY;
        else{
            float tt = (plane-si)/dirx;
            t_vals[cnt++] = (tt<0.f||tt>1.f)?INFINITY:tt;
        }
    }
    // y-planes
    for (int j=0;j<=n_y;++j){
        float plane = float(j)-0.5f;
        if (fabsf(diry)<1e-12f) t_vals[cnt++] = INFINITY;
        else{
            float tt = (plane-sj)/diry;
            t_vals[cnt++] = (tt<0.f||tt>1.f)?INFINITY:tt;
        }
    }
    // z-planes
    for (int k=0;k<=n_z;++k){
        float plane = float(k)-0.5f;
        if (fabsf(dirz)<1e-12f) t_vals[cnt++] = INFINITY;
        else{
            float tt = (plane-sk)/dirz;
            t_vals[cnt++] = (tt<0.f||tt>1.f)?INFINITY:tt;
        }
    }
    // bubble sort (unchanged)
    for (int i=0;i<n_int-1;++i)
        for (int j=0;j<n_int-1-i;++j)
            if (t_vals[j]>t_vals[j+1]){
                float tmp=t_vals[j]; t_vals[j]=t_vals[j+1]; t_vals[j+1]=tmp;
            }
}

// ……………………… forward_project_3d_kernel ………………………
__global__ void forward_project_3d_kernel(
    const float* volume,int batch,int n_x,int n_y,int n_z,
    const float* t_sorted,int n_int,
    const float* src_xyz,const float* dst_xyz,int n_ray,
    const float* M_inv,const float* b,
    float* out)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int total = batch*n_ray;
    if (gid>=total) return;

    int b_idx = gid / n_ray;
    int r_idx = gid % n_ray;

    float sx = src_xyz[3*r_idx+0], sy = src_xyz[3*r_idx+1], sz = src_xyz[3*r_idx+2];
    float dx = dst_xyz[3*r_idx+0], dy = dst_xyz[3*r_idx+1], dz = dst_xyz[3*r_idx+2];
    float vx=dx-sx, vy=dy-sy, vz=dz-sz;

    const float* t_vals = &t_sorted[r_idx*n_int];
    float accum = 0.f;

    for (int i=0;i<n_int-1;++i){
        float t0=t_vals[i], t1=t_vals[i+1];
        if (isinf(t0)||isinf(t1)) continue;

        float x0=sx+t0*vx, y0=sy+t0*vy, z0=sz+t0*vz;
        float x1=sx+t1*vx, y1=sy+t1*vy, z1=sz+t1*vz;
        float seg_len = sqrtf((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));

        float mx=0.5f*(x0+x1), my=0.5f*(y0+y1), mz=0.5f*(z0+z1);
        float i_f,j_f,k_f;
        apply_affine_inverse_3d(mx,my,mz, M_inv,b, i_f,j_f,k_f);

        int i_i = __float2int_rn(i_f);
        int j_i = __float2int_rn(j_f);
        int k_i = __float2int_rn(k_f);
        // float dirx = dx - sx, diry = dy - sy, dirz = dz - sz;
        // int i_i = (dirx >= 0.f) ? (int)floorf(i_f) : (int)ceilf(i_f) - 1;
        // int j_i = (diry >= 0.f) ? (int)floorf(j_f) : (int)ceilf(j_f) - 1;
        // int k_i = (dirz >= 0.f) ? (int)floorf(k_f) : (int)ceilf(k_f) - 1;

        if (i_i<0||i_i>=n_x||j_i<0||j_i>=n_y||k_i<0||k_i>=n_z) continue;

        size_t idx = (size_t)b_idx*n_x*n_y*n_z
                   + (size_t)i_i*n_y*n_z
                   + (size_t)j_i*n_z
                   + (size_t)k_i;

        accum += volume[idx]*seg_len;
    }
    out[b_idx*n_ray + r_idx] = accum;
}

// ……………………… back_project_3d_kernel ………………………
__global__ void back_project_3d_kernel(
    float* out_vol,int batch,int n_x,int n_y,int n_z,
    const float* t_sorted,int n_int,
    const float* sino,int n_ray,
    const float* src_xyz,const float* dst_xyz,
    const float* M_inv,const float* b)
{
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    int total = batch*n_ray;
    if (gid>=total) return;

    int b_idx = gid / n_ray;
    int r_idx = gid % n_ray;

    float sx=src_xyz[3*r_idx+0], sy=src_xyz[3*r_idx+1], sz=src_xyz[3*r_idx+2];
    float dx=dst_xyz[3*r_idx+0], dy=dst_xyz[3*r_idx+1], dz=dst_xyz[3*r_idx+2];
    float vx=dx-sx, vy=dy-sy, vz=dz-sz;

    const float* t_vals = &t_sorted[r_idx*n_int];
    float s_val = sino[b_idx*n_ray + r_idx];

    for (int i=0;i<n_int-1;++i){
        float t0=t_vals[i], t1=t_vals[i+1];
        if (isinf(t0)||isinf(t1)) continue;

        float x0=sx+t0*vx, y0=sy+t0*vy, z0=sz+t0*vz;
        float x1=sx+t1*vx, y1=sy+t1*vy, z1=sz+t1*vz;
        float seg_len = sqrtf((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0));

        float mx=0.5f*(x0+x1), my=0.5f*(y0+y1), mz=0.5f*(z0+z1);
        float i_f,j_f,k_f;
        apply_affine_inverse_3d(mx,my,mz, M_inv,b, i_f,j_f,k_f);

        int i_i = __float2int_rn(i_f);
        int j_i = __float2int_rn(j_f);
        int k_i = __float2int_rn(k_f);
        // float dirx = dx - sx, diry = dy - sy, dirz = dz - sz;
        // int i_i = (dirx >= 0.f) ? (int)floorf(i_f) : (int)ceilf(i_f) - 1;
        // int j_i = (diry >= 0.f) ? (int)floorf(j_f) : (int)ceilf(j_f) - 1;
        // int k_i = (dirz >= 0.f) ? (int)floorf(k_f) : (int)ceilf(k_f) - 1;

        if (i_i<0||i_i>=n_x||j_i<0||j_i>=n_y||k_i<0||k_i>=n_z) continue;

        float contrib = s_val*seg_len;
        size_t idx = (size_t)b_idx*n_x*n_y*n_z
                   + (size_t)i_i*n_y*n_z
                   + (size_t)j_i*n_z
                   + (size_t)k_i;
        atomicAdd(&out_vol[idx], contrib);
    }
}

// ------------------------------------------------------------
// HOST wrappers – signatures unchanged, but they now expect M_inv
// ------------------------------------------------------------
torch::Tensor compute_intersections_3d(
    int64_t n_x,int64_t n_y,int64_t n_z,
    torch::Tensor src,torch::Tensor dst,
    torch::Tensor M_inv,torch::Tensor b)
{
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA");
    TORCH_CHECK(M_inv.is_cuda() && b.is_cuda(), "M_inv,b must be CUDA");

    int64_t n_ray = src.size(0);
    int64_t n_int = (n_x+1)+(n_y+1)+(n_z+1);
    auto out = torch::empty({n_ray,n_int}, src.options().dtype(torch::kFloat32));

    int threads = 256;
    int blocks  = (int)((n_ray+threads-1)/threads);
    compute_intersections_3d_kernel<<<blocks,threads>>>(
        (int)n_x,(int)n_y,(int)n_z,
        src.data_ptr<float>(),dst.data_ptr<float>(),(int)n_ray,
        M_inv.data_ptr<float>(),b.data_ptr<float>(),
        out.data_ptr<float>());
    return out;
}

torch::Tensor forward_project_3d_cuda(
    torch::Tensor volume,torch::Tensor tvals,
    torch::Tensor src,torch::Tensor dst,
    torch::Tensor M_inv,torch::Tensor b)
{
    TORCH_CHECK(volume.is_cuda() && tvals.is_cuda(),"volume/tvals must be CUDA");

    int64_t batch=1,n_x,n_y,n_z;
    if (volume.dim()==3){n_x=volume.size(0);n_y=volume.size(1);n_z=volume.size(2);}
    else if (volume.dim()==4){
        batch=volume.size(0);n_x=volume.size(1);n_y=volume.size(2);n_z=volume.size(3);}
    else TORCH_CHECK(false,"volume shape!");

    int64_t n_ray = src.size(0);
    int64_t n_int = tvals.size(1);
    auto out = torch::zeros({batch,n_ray}, volume.options());

    int threads=256, blocks=(int)((batch*n_ray+threads-1)/threads);
    forward_project_3d_kernel<<<blocks,threads>>>(
        volume.data_ptr<float>(),(int)batch,(int)n_x,(int)n_y,(int)n_z,
        tvals.data_ptr<float>(),(int)n_int,
        src.data_ptr<float>(),dst.data_ptr<float>(),(int)n_ray,
        M_inv.data_ptr<float>(),b.data_ptr<float>(),
        out.data_ptr<float>());
    return out;
}

torch::Tensor back_project_3d_cuda(
    torch::Tensor sino,torch::Tensor tvals,
    torch::Tensor src,torch::Tensor dst,
    torch::Tensor M_inv,torch::Tensor b,
    int64_t n_x,int64_t n_y,int64_t n_z)
{
    TORCH_CHECK(sino.is_cuda() && tvals.is_cuda(),"sino/tvals must be CUDA");

    int64_t batch=1,n_ray;
    if (sino.dim()==1){n_ray=sino.size(0);}
    else if (sino.dim()==2){batch=sino.size(0);n_ray=sino.size(1);}
    else TORCH_CHECK(false,"sino shape!");

    int64_t n_int = tvals.size(1);
    auto out     = torch::zeros({batch,n_x,n_y,n_z}, sino.options());

    int threads=256, blocks=(int)((batch*n_ray+threads-1)/threads);
    back_project_3d_kernel<<<blocks,threads>>>(
        out.data_ptr<float>(),(int)batch,(int)n_x,(int)n_y,(int)n_z,
        tvals.data_ptr<float>(),(int)n_int,
        sino.data_ptr<float>(),(int)n_ray,
        src.data_ptr<float>(),dst.data_ptr<float>(),
        M_inv.data_ptr<float>(),b.data_ptr<float>());
    return out;
}












/* ========================================================================== */
/*               S I D D O N   O N - T H E - F L Y   K E R N E L S            */
/* ========================================================================== */

// ---------------------------------------------------------------------------
// Forward projector – one thread per (batch, ray) pair
// ---------------------------------------------------------------------------
__global__ void forward_project_3d_on_the_fly_kernel(
    const float* __restrict__ vol,        // [batch, X, Y, Z] or [X,Y,Z]
    int   batch, int n_x, int n_y, int n_z,
    const float* __restrict__ src_xyz,
    const float* __restrict__ dst_xyz,
    int   n_ray,
    const float* __restrict__ M_inv,
    const float* __restrict__ b,
    float* __restrict__ out)              // [batch, n_ray]
{
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (gid >= total) return;

    int b_idx = gid / n_ray;
    int r_idx = gid % n_ray;

    /* ---------- physical ray ------------------------------------------------ */
    float sx = src_xyz[3 * r_idx + 0];
    float sy = src_xyz[3 * r_idx + 1];
    float sz = src_xyz[3 * r_idx + 2];
    float dx = dst_xyz[3 * r_idx + 0];
    float dy = dst_xyz[3 * r_idx + 1];
    float dz = dst_xyz[3 * r_idx + 2];

    float vx = dx - sx, vy = dy - sy, vz = dz - sz;
    float ray_len = sqrtf(vx * vx + vy * vy + vz * vz);
    if (ray_len < 1e-12f) { out[gid] = 0.f; return; }

    /* ---------- map endpoints to voxel coordinates ------------------------- */
    float si, sj, sk, di, dj, dk;
    apply_affine_inverse_3d(sx, sy, sz, M_inv, b, si, sj, sk);
    apply_affine_inverse_3d(dx, dy, dz, M_inv, b, di, dj, dk);

    float vi = di - si, vj = dj - sj, vk = dk - sk;

    float inv_vi = (fabsf(vi) < 1e-12f) ? 1e32f : 1.f / vi;
    float inv_vj = (fabsf(vj) < 1e-12f) ? 1e32f : 1.f / vj;
    float inv_vk = (fabsf(vk) < 1e-12f) ? 1e32f : 1.f / vk;

    /* ---------- entry / exit parameters ------------------------------------ */
    float t_i0 = (-0.5f          - si) * inv_vi;
    float t_i1 = ((float)n_x-0.5f - si) * inv_vi;
    float t_j0 = (-0.5f          - sj) * inv_vj;
    float t_j1 = ((float)n_y-0.5f - sj) * inv_vj;
    float t_k0 = (-0.5f          - sk) * inv_vk;
    float t_k1 = ((float)n_z-0.5f - sk) * inv_vk;

    float t_enter = fmaxf(fmaxf(fminf(t_i0, t_i1), fminf(t_j0, t_j1)),
                          fminf(t_k0, t_k1));
    float t_exit  = fminf(fminf(fmaxf(t_i0, t_i1), fmaxf(t_j0, t_j1)),
                          fmaxf(t_k0, t_k1));
    if (t_exit <= t_enter) { out[gid] = 0.f; return; }

    t_enter = fmaxf(t_enter, 0.f);
    t_exit  = fminf(t_exit , 1.f);

    /* ---------- first voxel ------------------------------------------------- */
    float i_ent = si + t_enter * vi;
    float j_ent = sj + t_enter * vj;
    float k_ent = sk + t_enter * vk;

    int ii = (vi >= 0.f) ? (int)floorf(i_ent) : (int)ceilf(i_ent) - 1;
    int jj = (vj >= 0.f) ? (int)floorf(j_ent) : (int)ceilf(j_ent) - 1;
    int kk = (vk >= 0.f) ? (int)floorf(k_ent) : (int)ceilf(k_ent) - 1;

    float t_next_i = (vi >= 0.f)
        ? t_enter + ((ii + 0.5f) - i_ent) * inv_vi
        : t_enter + ((ii - 0.5f) - i_ent) * inv_vi;
    float t_next_j = (vj >= 0.f)
        ? t_enter + ((jj + 0.5f) - j_ent) * inv_vj
        : t_enter + ((jj - 0.5f) - j_ent) * inv_vj;
    float t_next_k = (vk >= 0.f)
        ? t_enter + ((kk + 0.5f) - k_ent) * inv_vk
        : t_enter + ((kk - 0.5f) - k_ent) * inv_vk;

    float dt_i = fabsf(inv_vi);
    float dt_j = fabsf(inv_vj);
    float dt_k = fabsf(inv_vk);

    /* ---------- voxel traversal ------------------------------------------- */
    float acc   = 0.f;
    float t_cur = t_enter;

    while (t_cur < t_exit)
    {
        float t_hit;
        int   axis;
        if      (t_next_i <= t_next_j && t_next_i <= t_next_k) { t_hit = t_next_i; axis = 0; }
        else if (t_next_j <= t_next_i && t_next_j <= t_next_k) { t_hit = t_next_j; axis = 1; }
        else                                                   { t_hit = t_next_k; axis = 2; }

        t_hit = fminf(t_hit, t_exit);

        if (ii >= 0 && ii < n_x &&
            jj >= 0 && jj < n_y &&
            kk >= 0 && kk < n_z)
        {
            float seg = (t_hit - t_cur) * ray_len;
            size_t idx = (size_t)b_idx * n_x * n_y * n_z
                       + (size_t)ii    * n_y * n_z
                       + (size_t)jj    * n_z
                       + (size_t)kk;
            acc += vol[idx] * seg;
        }

        t_cur = t_hit;
        if      (axis == 0) { t_next_i += dt_i; ii += (vi >= 0.f) ? 1 : -1; }
        else if (axis == 1) { t_next_j += dt_j; jj += (vj >= 0.f) ? 1 : -1; }
        else                { t_next_k += dt_k; kk += (vk >= 0.f) ? 1 : -1; }
    }
    out[gid] = acc;
}

// ---------------------------------------------------------------------------
// Back-projector – adjoint of the forward kernel
// ---------------------------------------------------------------------------
__global__ void back_project_3d_on_the_fly_kernel(
    float* __restrict__ vol_out,          // [batch,X,Y,Z]
    int   batch, int n_x, int n_y, int n_z,
    const float* __restrict__ sino,
    int   n_ray,
    const float* __restrict__ src_xyz,
    const float* __restrict__ dst_xyz,
    const float* __restrict__ M_inv,
    const float* __restrict__ b)
{
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (gid >= total) return;

    int b_idx = gid / n_ray;
    int r_idx = gid % n_ray;

    /* ---------- physical ray ---------------------------------------------- */
    float sx = src_xyz[3 * r_idx + 0];
    float sy = src_xyz[3 * r_idx + 1];
    float sz = src_xyz[3 * r_idx + 2];
    float dx = dst_xyz[3 * r_idx + 0];
    float dy = dst_xyz[3 * r_idx + 1];
    float dz = dst_xyz[3 * r_idx + 2];

    float vx = dx - sx, vy = dy - sy, vz = dz - sz;
    float ray_len = sqrtf(vx * vx + vy * vy + vz * vz);
    if (ray_len < 1e-12f) return;

    /* ---------- voxel-space endpoints ------------------------------------- */
    float si, sj, sk, di, dj, dk;
    apply_affine_inverse_3d(sx, sy, sz, M_inv, b, si, sj, sk);
    apply_affine_inverse_3d(dx, dy, dz, M_inv, b, di, dj, dk);

    float vi = di - si, vj = dj - sj, vk = dk - sk;

    float inv_vi = (fabsf(vi) < 1e-12f) ? 1e32f : 1.f / vi;
    float inv_vj = (fabsf(vj) < 1e-12f) ? 1e32f : 1.f / vj;
    float inv_vk = (fabsf(vk) < 1e-12f) ? 1e32f : 1.f / vk;

    float t_i0 = (-0.5f          - si) * inv_vi;
    float t_i1 = ((float)n_x-0.5f - si) * inv_vi;
    float t_j0 = (-0.5f          - sj) * inv_vj;
    float t_j1 = ((float)n_y-0.5f - sj) * inv_vj;
    float t_k0 = (-0.5f          - sk) * inv_vk;
    float t_k1 = ((float)n_z-0.5f - sk) * inv_vk;

    float t_enter = fmaxf(fmaxf(fminf(t_i0, t_i1), fminf(t_j0, t_j1)),
                          fminf(t_k0, t_k1));
    float t_exit  = fminf(fminf(fmaxf(t_i0, t_i1), fmaxf(t_j0, t_j1)),
                          fmaxf(t_k0, t_k1));
    if (t_exit <= t_enter) return;

    t_enter = fmaxf(t_enter, 0.f);
    t_exit  = fminf(t_exit , 1.f);

    /* ---------- first voxel ------------------------------------------------ */
    float i_ent = si + t_enter * vi;
    float j_ent = sj + t_enter * vj;
    float k_ent = sk + t_enter * vk;

    int ii = (vi >= 0.f) ? (int)floorf(i_ent) : (int)ceilf(i_ent) - 1;
    int jj = (vj >= 0.f) ? (int)floorf(j_ent) : (int)ceilf(j_ent) - 1;
    int kk = (vk >= 0.f) ? (int)floorf(k_ent) : (int)ceilf(k_ent) - 1;

    float t_next_i = (vi >= 0.f)
        ? t_enter + ((ii + 0.5f) - i_ent) * inv_vi
        : t_enter + ((ii - 0.5f) - i_ent) * inv_vi;
    float t_next_j = (vj >= 0.f)
        ? t_enter + ((jj + 0.5f) - j_ent) * inv_vj
        : t_enter + ((jj - 0.5f) - j_ent) * inv_vj;
    float t_next_k = (vk >= 0.f)
        ? t_enter + ((kk + 0.5f) - k_ent) * inv_vk
        : t_enter + ((kk - 0.5f) - k_ent) * inv_vk;

    float dt_i = fabsf(inv_vi);
    float dt_j = fabsf(inv_vj);
    float dt_k = fabsf(inv_vk);

    float proj_val = sino[gid];
    float t_cur    = t_enter;

    /* ---------- voxel traversal ------------------------------------------- */
    while (t_cur < t_exit)
    {
        float t_hit;
        int   axis;
        if      (t_next_i <= t_next_j && t_next_i <= t_next_k) { t_hit = t_next_i; axis = 0; }
        else if (t_next_j <= t_next_i && t_next_j <= t_next_k) { t_hit = t_next_j; axis = 1; }
        else                                                   { t_hit = t_next_k; axis = 2; }

        t_hit = fminf(t_hit, t_exit);

        if (ii >= 0 && ii < n_x &&
            jj >= 0 && jj < n_y &&
            kk >= 0 && kk < n_z)
        {
            float seg = (t_hit - t_cur) * ray_len;
            size_t idx = (size_t)b_idx * n_x * n_y * n_z
                       + (size_t)ii    * n_y * n_z
                       + (size_t)jj    * n_z
                       + (size_t)kk;
            atomicAdd(&vol_out[idx], proj_val * seg);
        }

        t_cur = t_hit;
        if      (axis == 0) { t_next_i += dt_i; ii += (vi >= 0.f) ? 1 : -1; }
        else if (axis == 1) { t_next_j += dt_j; jj += (vj >= 0.f) ? 1 : -1; }
        else                { t_next_k += dt_k; kk += (vk >= 0.f) ? 1 : -1; }
    }
}

/* ────────────────────────────────────────────────────────────────────────── */
/*           H O S T   W R A P P E R S   (Siddon on-the-fly)                 */
/* ────────────────────────────────────────────────────────────────────────── */

torch::Tensor forward_project_3d_on_the_fly_cuda(
    torch::Tensor volume,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M_inv,
    torch::Tensor b)
{
    TORCH_CHECK(volume.is_cuda() && src.is_cuda() && dst.is_cuda(),
                "inputs must be CUDA tensors");

    int64_t batch = 1, n_x, n_y, n_z;
    if (volume.dim() == 3) {
        n_x = volume.size(0);
        n_y = volume.size(1);
        n_z = volume.size(2);
    } else if (volume.dim() == 4) {
        batch = volume.size(0);
        n_x   = volume.size(1);
        n_y   = volume.size(2);
        n_z   = volume.size(3);
    } else {
        TORCH_CHECK(false, "volume must be 3-D or 4-D");
    }

    int64_t n_ray = src.size(0);
    auto out = torch::zeros({batch, n_ray}, volume.options());

    int threads = 256;
    int blocks  = static_cast<int>((batch * n_ray + threads - 1) / threads);

    forward_project_3d_on_the_fly_kernel<<<blocks, threads>>>(
        volume.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(n_x),
        static_cast<int>(n_y),
        static_cast<int>(n_z),
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        static_cast<int>(n_ray),
        M_inv.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>());

    return out;
}

torch::Tensor back_project_3d_on_the_fly_cuda(
    torch::Tensor sino,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M_inv,
    torch::Tensor b,
    int64_t n_x,
    int64_t n_y,
    int64_t n_z)
{
    TORCH_CHECK(sino.is_cuda() && src.is_cuda() && dst.is_cuda(),
                "inputs must be CUDA tensors");

    int64_t batch = 1, n_ray;
    if (sino.dim() == 1) {
        n_ray = sino.size(0);
    } else if (sino.dim() == 2) {
        batch = sino.size(0);
        n_ray = sino.size(1);
    } else {
        TORCH_CHECK(false, "sinogram must be 1-D or 2-D");
    }

    auto out = torch::zeros({batch, n_x, n_y, n_z}, sino.options());

    int threads = 256;
    int blocks  = static_cast<int>((batch * n_ray + threads - 1) / threads);

    back_project_3d_on_the_fly_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(n_x),
        static_cast<int>(n_y),
        static_cast<int>(n_z),
        sino.data_ptr<float>(),
        static_cast<int>(n_ray),
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        M_inv.data_ptr<float>(),
        b.data_ptr<float>());

    return out;
}
