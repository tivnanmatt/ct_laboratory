// File: src/ct_projector_3d.cu

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------------
// HELPER: apply_affine_inverse_3d(x, y, z) = A_inv * ((x,y,z) - b) => (i, j, k)
//   A_inv has shape [3,3], stored as 9 floats: 
//     [a11, a12, a13,  a21, a22, a23,  a31, a32, a33]
//   b has shape [3]
// ---------------------------------------------------------------------------------
__device__ __forceinline__
void apply_affine_inverse_3d(
    float x, float y, float z,
    const float* M, // [9] (input matrix, will be inverted inside)
    const float* b, // [3]
    float& i_out, float& j_out, float& k_out
) {
    // Compute inverse of M (3x3) on the fly.
    float m0 = M[0], m1 = M[1], m2 = M[2];
    float m3 = M[3], m4 = M[4], m5 = M[5];
    float m6 = M[6], m7 = M[7], m8 = M[8];
    float det = m0*(m4*m8 - m5*m7) - m1*(m3*m8 - m5*m6) + m2*(m3*m7 - m4*m6);
    float inv_det = 1.f / det;
    float a11 = (m4*m8 - m5*m7) * inv_det;
    float a12 = (m2*m7 - m1*m8) * inv_det;
    float a13 = (m1*m5 - m2*m4) * inv_det;
    float a21 = (m5*m6 - m3*m8) * inv_det;
    float a22 = (m0*m8 - m2*m6) * inv_det;
    float a23 = (m2*m3 - m0*m5) * inv_det;
    float a31 = (m3*m7 - m4*m6) * inv_det;
    float a32 = (m1*m6 - m0*m7) * inv_det;
    float a33 = (m0*m4 - m1*m3) * inv_det;

    float x_shift = x - b[0];
    float y_shift = y - b[1];
    float z_shift = z - b[2];
    // Layout A_inv: [0]=a11, [1]=a12, [2]=a13, [3]=a21, [4]=a22, [5]=a23, [6]=a31, [7]=a32, [8]=a33
    i_out = a11*x_shift + a12*y_shift + a13*z_shift;
    j_out = a21*x_shift + a22*y_shift + a23*z_shift;
    k_out = a31*x_shift + a32*y_shift + a33*z_shift;
}

// ---------------------------------------------------------------------------------
// KERNEL 1: compute_intersections_3d_kernel
//
//   For each ray r in [0..n_ray-1], we compute intersection parameters t
//   for x-planes => i-0.5, i=0..n_x
//       y-planes => j-0.5, j=0..n_y
//       z-planes => k-0.5, k=0..n_z
//   (in "index" space (i,j,k)) after applying A_inv,b.
//
//   Filter t<0 or t>1 => +inf, then do a simple bubble sort ascending.
//
//   Output: t_out[r, 0..(n_intersections-1)]   (float)
//           where n_intersections = (n_x+1 + n_y+1 + n_z+1).
// ---------------------------------------------------------------------------------
__global__ void compute_intersections_3d_kernel(
    int n_x,
    int n_y,
    int n_z,

    const float* src_xyz, // [n_ray, 3]
    const float* dst_xyz, // [n_ray, 3]
    int n_ray,

    const float* M,   // [9]
    const float* b,   // [3]

    // Output => [n_ray, n_intersections]
    float* t_out
) {
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= n_ray) return;

    // read src, dst
    float sx = src_xyz[ray_id*3 + 0];
    float sy = src_xyz[ray_id*3 + 1];
    float sz = src_xyz[ray_id*3 + 2];
    float dx = dst_xyz[ray_id*3 + 0];
    float dy = dst_xyz[ray_id*3 + 1];
    float dz = dst_xyz[ray_id*3 + 2];

    // Transform (sx,sy,sz)->(si,sj,sk), (dx,dy,dz)->(di,dj,dk)
    float si, sj, sk, di, dj, dk;
    apply_affine_inverse_3d(sx, sy, sz, M, b, si, sj, sk);
    apply_affine_inverse_3d(dx, dy, dz, M, b, di, dj, dk);

    float dirx = di - si;
    float diry = dj - sj;
    float dirz = dk - sk;

    int n_intersections = (n_x + 1) + (n_y + 1) + (n_z + 1);

    // pointer to the row in t_out for this ray
    float* t_vals = &t_out[ray_id * n_intersections];

    // 1) x-planes => i=0..n_x => i - 0.5
    int count = 0;
    for (int i=0; i <= n_x; i++) {
        float plane_i = float(i) - 0.5f;
        if (fabsf(dirx) < 1e-12f) {
            t_vals[count++] = INFINITY;
        } else {
            float tt = (plane_i - si) / dirx;
            if (tt < 0.f || tt > 1.f) {
                t_vals[count++] = INFINITY;
            } else {
                t_vals[count++] = tt;
            }
        }
    }

    // 2) y-planes => j=0..n_y => j - 0.5
    for (int j=0; j <= n_y; j++) {
        float plane_j = float(j) - 0.5f;
        if (fabsf(diry) < 1e-12f) {
            t_vals[count++] = INFINITY;
        } else {
            float tt = (plane_j - sj) / diry;
            if (tt < 0.f || tt > 1.f) {
                t_vals[count++] = INFINITY;
            } else {
                t_vals[count++] = tt;
            }
        }
    }

    // 3) z-planes => k=0..n_z => k - 0.5
    for (int k=0; k <= n_z; k++) {
        float plane_k = float(k) - 0.5f;
        if (fabsf(dirz) < 1e-12f) {
            t_vals[count++] = INFINITY;
        } else {
            float tt = (plane_k - sk) / dirz;
            if (tt < 0.f || tt > 1.f) {
                t_vals[count++] = INFINITY;
            } else {
                t_vals[count++] = tt;
            }
        }
    }

    // bubble sort of t_vals
    for (int i=0; i < n_intersections-1; i++) {
        for (int j=0; j < n_intersections-1-i; j++) {
            if (t_vals[j] > t_vals[j+1]) {
                float tmp = t_vals[j];
                t_vals[j] = t_vals[j+1];
                t_vals[j+1] = tmp;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// KERNEL 2: forward_project_3d_kernel
//   - Takes intersections t_sorted[r, i], plus src_xyz, dst_xyz, and volume
//   - For each (batch, ray), we accumulate the line integral in 3D
// ---------------------------------------------------------------------------------
__global__ void forward_project_3d_kernel(
    // volume => [batch, n_x, n_y, n_z] or [n_x, n_y, n_z] if batch=1
    const float* volume,
    int batch,
    int n_x,
    int n_y,
    int n_z,

    // t_sorted => [n_ray, n_intersections], n_intersections = (n_x+1 + n_y+1 + n_z+1)
    const float* t_sorted,
    int n_intersections,

    // geometry
    const float* src_xyz, // [n_ray,3]
    const float* dst_xyz, // [n_ray,3]
    int n_ray,

    const float* M, // [9]
    const float* b, // [3]

    // output => [batch, n_ray]
    float* out
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (global_id >= total) return;

    int b_idx = global_id / n_ray;  // batch index
    int r_idx = global_id % n_ray;  // ray index

    // read src, dst
    float sx = src_xyz[r_idx*3 + 0];
    float sy = src_xyz[r_idx*3 + 1];
    float sz = src_xyz[r_idx*3 + 2];
    float dx = dst_xyz[r_idx*3 + 0];
    float dy = dst_xyz[r_idx*3 + 1];
    float dz = dst_xyz[r_idx*3 + 2];

    float vx = dx - sx;
    float vy = dy - sy;
    float vz = dz - sz;

    // pointer to the t-values for this ray => t_sorted[r_idx, ...]
    const float* t_vals = &t_sorted[r_idx * n_intersections];

    float accum_val = 0.f;
    for (int i=0; i < n_intersections - 1; i++) {
        float t0 = t_vals[i];
        float t1 = t_vals[i+1];
        if (isinf(t0) || isinf(t1)) {
            continue;
        }
        // segment endpoints
        float x0 = sx + t0*vx;
        float y0 = sy + t0*vy;
        float z0 = sz + t0*vz;
        float x1 = sx + t1*vx;
        float y1 = sy + t1*vy;
        float z1 = sz + t1*vz;

        float dx_ = (x1 - x0);
        float dy_ = (y1 - y0);
        float dz_ = (z1 - z0);
        float seg_len = sqrtf(dx_*dx_ + dy_*dy_ + dz_*dz_);

        // midpoint
        float mx = 0.5f*(x0 + x1);
        float my = 0.5f*(y0 + y1);
        float mz = 0.5f*(z0 + z1);

        // transform => i, j, k
        float i_f, j_f, k_f;
        apply_affine_inverse_3d(mx, my, mz, M, b, i_f, j_f, k_f);
        int i_i = (int) roundf(i_f);
        int j_i = (int) roundf(j_f);
        int k_i = (int) roundf(k_f);

        if (i_i < 0 || i_i >= n_x || j_i < 0 || j_i >= n_y || k_i < 0 || k_i >= n_z) {
            continue;
        }

        // volume index
        // layout: batch*(n_x*n_y*n_z) + i*(n_y*n_z) + j*(n_z) + k
        size_t idx_vol = b_idx*( (size_t)n_x*n_y*n_z )
                       + (size_t)i_i*( (size_t)n_y*n_z )
                       + (size_t)j_i*( (size_t)n_z )
                       + (size_t)k_i;

        float voxel_val = volume[idx_vol];
        accum_val += voxel_val * seg_len;
    }

    out[b_idx*n_ray + r_idx] = accum_val;
}

// ---------------------------------------------------------------------------------
// HELPER for atomicAdd on float if needed for older arch
// ---------------------------------------------------------------------------------
__device__ __forceinline__
void atomicAddFloat3D(float* address, float val) {
#if __CUDA_ARCH__ < 600
    atomicAdd(address, val);
#else
    atomicAdd(address, val);
#endif
}

// ---------------------------------------------------------------------------------
// KERNEL 3: back_project_3d_kernel
//   - Takes intersections t_sorted[r, i], plus sinogram => scatter-add into volume
// ---------------------------------------------------------------------------------
__global__ void back_project_3d_kernel(
    // out_volume => [batch, n_x, n_y, n_z]
    float* out_volume,
    int batch,
    int n_x,
    int n_y,
    int n_z,

    // intersections => [n_ray, n_intersections]
    const float* t_sorted,
    int n_intersections,

    // sinogram => [batch, n_ray] or [n_ray] if batch=1
    const float* sinogram,
    int n_ray,

    // geometry
    const float* src_xyz, // [n_ray,3]
    const float* dst_xyz, // [n_ray,3]
    const float* M,       // [9]
    const float* b        // [3]
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (global_id >= total) return;

    int b_idx = global_id / n_ray;
    int r_idx = global_id % n_ray;

    float sx = src_xyz[r_idx*3 + 0];
    float sy = src_xyz[r_idx*3 + 1];
    float sz = src_xyz[r_idx*3 + 2];
    float dx = dst_xyz[r_idx*3 + 0];
    float dy = dst_xyz[r_idx*3 + 1];
    float dz = dst_xyz[r_idx*3 + 2];

    float vx = dx - sx;
    float vy = dy - sy;
    float vz = dz - sz;

    const float* t_vals = &t_sorted[r_idx * n_intersections];

    // sinogram value
    float s_val = sinogram[b_idx*n_ray + r_idx];

    for (int i=0; i < n_intersections - 1; i++) {
        float t0 = t_vals[i];
        float t1 = t_vals[i+1];
        if (isinf(t0) || isinf(t1)) {
            continue;
        }
        float x0 = sx + t0*vx;
        float y0 = sy + t0*vy;
        float z0 = sz + t0*vz;
        float x1 = sx + t1*vx;
        float y1 = sy + t1*vy;
        float z1 = sz + t1*vz;

        float dx_ = (x1 - x0);
        float dy_ = (y1 - y0);
        float dz_ = (z1 - z0);
        float seg_len = sqrtf(dx_*dx_ + dy_*dy_ + dz_*dz_);

        // midpoint
        float mx = 0.5f*(x0 + x1);
        float my = 0.5f*(y0 + y1);
        float mz = 0.5f*(z0 + z1);

        float i_f, j_f, k_f;
        apply_affine_inverse_3d(mx, my, mz, M, b, i_f, j_f, k_f);
        int i_i = (int) roundf(i_f);
        int j_i = (int) roundf(j_f);
        int k_i = (int) roundf(k_f);

        if (i_i<0 || i_i>=n_x || j_i<0 || j_i>=n_y || k_i<0 || k_i>=n_z) {
            continue;
        }

        float contrib = s_val * seg_len;
        size_t idx_vol = b_idx*( (size_t)n_x*n_y*n_z )
                       + (size_t)i_i*( (size_t)n_y*n_z )
                       + (size_t)j_i*( (size_t)n_z )
                       + (size_t)k_i;

        atomicAddFloat3D(&out_volume[idx_vol], contrib);
    }
}

// ---------------------------------------------------------------------------------
// PUBLIC FUNCTIONS (C++), callable via PyBind11
// ---------------------------------------------------------------------------------

// 1) compute_intersections_3d
torch::Tensor compute_intersections_3d(
    int64_t n_x,
    int64_t n_y,
    int64_t n_z,

    torch::Tensor src,   // [n_ray,3]
    torch::Tensor dst,   // [n_ray,3]
    torch::Tensor M,     // [3,3]
    torch::Tensor b      // [3]
) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA");
    TORCH_CHECK(M.is_cuda() && b.is_cuda(), "M, b must be CUDA");

    int64_t n_ray = src.size(0);
    int64_t n_intersections = (n_x + 1) + (n_y + 1) + (n_z + 1);

    auto out_options = src.options().dtype(torch::kFloat32);
    auto tvals = torch::empty({n_ray, n_intersections}, out_options);

    // Launch 1D kernel => one thread per ray
    int threads = 256;
    int blocks = (int)((n_ray + threads - 1)/threads);

    compute_intersections_3d_kernel<<<blocks, threads>>>(
        (int)n_x, (int)n_y, (int)n_z,
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        (int)n_ray,
        M.data_ptr<float>(),
        b.data_ptr<float>(),
        tvals.data_ptr<float>()
    );

    return tvals;
}

// 2) forward_project_3d_cuda
torch::Tensor forward_project_3d_cuda(
    torch::Tensor volume,     // [B, n_x, n_y, n_z] or [n_x,n_y,n_z]
    torch::Tensor tvals,      // [n_ray, n_intersections]
    torch::Tensor src,        // [n_ray,3]
    torch::Tensor dst,        // [n_ray,3]
    torch::Tensor M,          // [3,3]
    torch::Tensor b           // [3]
) {
    TORCH_CHECK(volume.is_cuda(), "volume must be CUDA");
    TORCH_CHECK(tvals.is_cuda(),  "tvals must be CUDA");

    int64_t batch = 1;
    int64_t n_x, n_y, n_z;
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
        TORCH_CHECK(false, "volume must be [n_x,n_y,n_z] or [B,n_x,n_y,n_z]");
    }

    int64_t n_ray = src.size(0);
    int64_t n_intersections = tvals.size(1);

    // output => [batch, n_ray]
    auto out = torch::zeros({batch, n_ray}, volume.options());

    // Launch kernel
    int threads = 256;
    int blocks = (int)((batch * n_ray + threads - 1)/threads);

    forward_project_3d_kernel<<<blocks, threads>>>(
        volume.data_ptr<float>(),
        (int)batch, (int)n_x, (int)n_y, (int)n_z,
        tvals.data_ptr<float>(),
        (int)n_intersections,
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        (int)n_ray,
        M.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>()
    );
    return out;
}

// 3) back_project_3d_cuda
torch::Tensor back_project_3d_cuda(
    torch::Tensor sinogram, // [B,n_ray] or [n_ray]
    torch::Tensor tvals,    // [n_ray, n_intersections]
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b,
    int64_t n_x,
    int64_t n_y,
    int64_t n_z
) {
    TORCH_CHECK(sinogram.is_cuda() && tvals.is_cuda(), "sinogram, tvals must be CUDA");

    int64_t batch = 1, n_ray = 0;
    if (sinogram.dim() == 1) {
        n_ray = sinogram.size(0);
    } else if (sinogram.dim() == 2) {
        batch = sinogram.size(0);
        n_ray = sinogram.size(1);
    } else {
        TORCH_CHECK(false, "sinogram must be [n_ray] or [B,n_ray]");
    }
    int64_t n_intersections = tvals.size(1);

    // out_volume => [batch, n_x, n_y, n_z]
    auto out_volume = torch::zeros({batch, n_x, n_y, n_z}, sinogram.options());

    // Launch kernel
    int threads = 256;
    int blocks = (int)((batch*n_ray + threads - 1)/threads);

    back_project_3d_kernel<<<blocks, threads>>>(
        out_volume.data_ptr<float>(),
        (int)batch, (int)n_x, (int)n_y, (int)n_z,
        tvals.data_ptr<float>(),
        (int)n_intersections,
        sinogram.data_ptr<float>(),
        (int)n_ray,
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        M.data_ptr<float>(),
        b.data_ptr<float>()
    );

    return out_volume;
}
