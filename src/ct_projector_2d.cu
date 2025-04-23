// File: src/ct_projector_2d.cu

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------------
// HELPER: apply_affine_inverse(x, y) = M_inv * ((x,y) - b) => (row, col)
//   M has shape [2,2], stored as 4 floats: [m11, m12, m21, m22]
//   b has shape [2]
// ---------------------------------------------------------------------------------
__device__ __forceinline__
void apply_affine_inverse(
    float x, float y,
    const float* M,  // [4]
    const float* b,  // [2]
    float& row, float& col
) {
    float x_shift = x - b[0];
    float y_shift = y - b[1];

    // Compute the determinant of M
    float det = M[0] * M[3] - M[1] * M[2];
    if (fabsf(det) < 1e-12f) {
        // Singular matrix, return NaN
        row = NAN;
        col = NAN;
        return;
    }
    
    // Compute the inverse manually
    float inv_m11 = M[3] / det;
    float inv_m12 = -M[1] / det;
    float inv_m21 = -M[2] / det;
    float inv_m22 = M[0] / det;

    // Apply inverse transformation
    row = inv_m11 * x_shift + inv_m12 * y_shift;
    col = inv_m21 * x_shift + inv_m22 * y_shift;
}

// ---------------------------------------------------------------------------------
// KERNEL 1: compute_intersections_2d_kernel
// ---------------------------------------------------------------------------------
__global__ void compute_intersections_2d_kernel(
    int n_row,
    int n_col,

    const float* src_xy,  // [n_ray, 2]
    const float* dst_xy,  // [n_ray, 2]
    int n_ray,

    const float* M,  // [4]
    const float* b,  // [2]

    float* t_out  // Output => [n_ray, n_intersections]
) {
    int ray_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_id >= n_ray) return;

    float sx = src_xy[ray_id*2 + 0];
    float sy = src_xy[ray_id*2 + 1];
    float dx = dst_xy[ray_id*2 + 0];
    float dy = dst_xy[ray_id*2 + 1];

    float srx, sry, drx, dry;
    apply_affine_inverse(sx, sy, M, b, srx, sry);
    apply_affine_inverse(dx, dy, M, b, drx, dry);

    float dirx = drx - srx;
    float diry = dry - sry;

    int n_intersections = (n_row + 1) + (n_col + 1);
    float* t_vals = &t_out[ray_id * n_intersections];

    int count = 0;
    for (int i=0; i<=n_row; i++) {
        float rc = float(i) - 0.5f;
        if (fabsf(dirx) < 1e-12f) {
            t_vals[count++] = INFINITY;
        } else {
            float tt = (rc - srx) / dirx;
            t_vals[count++] = (tt < 0.f || tt > 1.f) ? INFINITY : tt;
        }
    }
    for (int j=0; j<=n_col; j++) {
        float cc = float(j) - 0.5f;
        if (fabsf(diry) < 1e-12f) {
            t_vals[count++] = INFINITY;
        } else {
            float tt = (cc - sry) / diry;
            t_vals[count++] = (tt < 0.f || tt > 1.f) ? INFINITY : tt;
        }
    }

    for (int i=0; i<n_intersections-1; i++) {
        for (int j=0; j<n_intersections-1-i; j++) {
            if (t_vals[j] > t_vals[j+1]) {
                float tmp = t_vals[j];
                t_vals[j] = t_vals[j+1];
                t_vals[j+1] = tmp;
            }
        }
    }
}

// ---------------------------------------------------------------------------------
// KERNEL 2: forward_project_2d_kernel
//   - Takes intersections t_sorted[r, i], plus src_xy, dst_xy, and image
//   - For each (batch, ray), we accumulate the line integral
// ---------------------------------------------------------------------------------
__global__ void forward_project_2d_kernel(
    const float* image,
    int batch,
    int n_row,
    int n_col,

    const float* t_sorted,
    int n_intersections,

    const float* src_xy, 
    const float* dst_xy,
    int n_ray,

    const float* M, 
    const float* b,

    float* out
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (global_id >= total) return;

    int b_idx = global_id / n_ray;
    int r_idx = global_id % n_ray;

    float sx = src_xy[r_idx * 2 + 0];
    float sy = src_xy[r_idx * 2 + 1];
    float dx = dst_xy[r_idx * 2 + 0];
    float dy = dst_xy[r_idx * 2 + 1];

    float vx = dx - sx;
    float vy = dy - sy;

    const float* t_vals = &t_sorted[r_idx * n_intersections];

    float accum_val = 0.f;
    for (int i = 0; i < n_intersections - 1; i++) {
        float t0 = t_vals[i];
        float t1 = t_vals[i + 1];
        if (isinf(t0) || isinf(t1)) {
            continue;
        }

        float x0 = sx + t0 * vx;
        float y0 = sy + t0 * vy;
        float x1 = sx + t1 * vx;
        float y1 = sy + t1 * vy;
        float seg_len = sqrtf((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));

        float mx = 0.5f * (x0 + x1);
        float my = 0.5f * (y0 + y1);

        float rowf, colf;
        apply_affine_inverse(mx, my, M, b, rowf, colf);
        int rowi = (int) roundf(rowf);
        int coli = (int) roundf(colf);

        if (rowi < 0 || rowi >= n_row || coli < 0 || coli >= n_col) {
            continue;
        }

        int idx_img = b_idx * (n_row * n_col) + rowi * n_col + coli;
        float pixel_val = image[idx_img];
        accum_val += pixel_val * seg_len;
    }

    out[b_idx * n_ray + r_idx] = accum_val;
}

// ---------------------------------------------------------------------------------
// KERNEL 3: back_project_2d_kernel
//   - Takes intersections t_sorted[r, i], plus sinogram => scatter-add into image
// ---------------------------------------------------------------------------------
__global__ void back_project_2d_kernel(
    float* out_image,
    int batch,
    int n_row,
    int n_col,

    const float* t_sorted,
    int n_intersections,

    const float* sinogram,
    int n_ray,

    const float* src_xy,
    const float* dst_xy,
    const float* M,
    const float* b
) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (global_id >= total) return;

    int b_idx = global_id / n_ray;
    int r_idx = global_id % n_ray;

    float sx = src_xy[r_idx * 2 + 0];
    float sy = src_xy[r_idx * 2 + 1];
    float dx = dst_xy[r_idx * 2 + 0];
    float dy = dst_xy[r_idx * 2 + 1];

    float vx = dx - sx;
    float vy = dy - sy;

    const float* t_vals = &t_sorted[r_idx * n_intersections];

    float s_val = sinogram[b_idx * n_ray + r_idx];

    for (int i = 0; i < n_intersections - 1; i++) {
        float t0 = t_vals[i];
        float t1 = t_vals[i + 1];
        if (isinf(t0) || isinf(t1)) {
            continue;
        }

        float x0 = sx + t0 * vx;
        float y0 = sy + t0 * vy;
        float x1 = sx + t1 * vx;
        float y1 = sy + t1 * vy;
        float seg_len = sqrtf((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));

        float mx = 0.5f * (x0 + x1);
        float my = 0.5f * (y0 + y1);

        float rowf, colf;
        apply_affine_inverse(mx, my, M, b, rowf, colf);
        int rowi = (int) roundf(rowf);
        int coli = (int) roundf(colf);

        if (rowi < 0 || rowi >= n_row || coli < 0 || coli >= n_col) {
            continue;
        }

        float contrib = s_val * seg_len;
        int idx_img = b_idx * (n_row * n_col) + rowi * n_col + coli;

        atomicAdd(&out_image[idx_img], contrib);
    }
}

// ---------------------------------------------------------------------------------
// PUBLIC FUNCTIONS (C++), callable via PyBind11
// ---------------------------------------------------------------------------------

torch::Tensor compute_intersections_2d(
    int64_t n_row,
    int64_t n_col,

    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b
) {
    TORCH_CHECK(src.is_cuda() && dst.is_cuda(), "src/dst must be CUDA");
    TORCH_CHECK(M.is_cuda() && b.is_cuda(), "M, b must be CUDA");
    int64_t n_ray = src.size(0);
    int64_t n_intersections = (n_row + 1) + (n_col + 1);

    auto out_options = src.options().dtype(torch::kFloat32);
    auto tvals = torch::empty({n_ray, n_intersections}, out_options);

    int threads = 256;
    int blocks = (int)((n_ray + threads-1)/threads);

    compute_intersections_2d_kernel<<<blocks, threads>>>(
        (int)n_row, (int)n_col,
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        (int)n_ray,
        M.data_ptr<float>(),
        b.data_ptr<float>(),
        tvals.data_ptr<float>()
    );
    return tvals;
}

torch::Tensor forward_project_2d_cuda(
    torch::Tensor image,      
    torch::Tensor tvals,      
    torch::Tensor src,        
    torch::Tensor dst,        
    torch::Tensor M,          
    torch::Tensor b           
) {
    TORCH_CHECK(image.is_cuda(), "image must be CUDA");
    TORCH_CHECK(tvals.is_cuda(), "tvals must be CUDA");
    
    int64_t batch=1, n_row, n_col;
    if (image.dim()==2) {
        n_row = image.size(0);
        n_col = image.size(1);
    } else if (image.dim()==3) {
        batch = image.size(0);
        n_row = image.size(1);
        n_col = image.size(2);
    } else {
        TORCH_CHECK(false, "image must be [R,C] or [B,R,C]");
    }

    int64_t n_ray = src.size(0);
    int64_t n_intersections = tvals.size(1);

    auto out = torch::zeros({batch, n_ray}, image.options());

    int threads = 256;
    int blocks = (int)((batch*n_ray + threads-1)/threads);

    forward_project_2d_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        (int)batch, (int)n_row, (int)n_col,
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

torch::Tensor back_project_2d_cuda(
    torch::Tensor sinogram, 
    torch::Tensor tvals,    
    torch::Tensor src, 
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b,
    int64_t n_row,
    int64_t n_col
) {
    int64_t batch=1, n_ray;
    if (sinogram.dim()==1) {
        n_ray = sinogram.size(0);
    } else if (sinogram.dim()==2) {
        batch = sinogram.size(0);
        n_ray = sinogram.size(1);
    } else {
        TORCH_CHECK(false, "sinogram must be [n_ray] or [B,n_ray]");
    }

    int64_t n_intersections = tvals.size(1);

    auto out_image = torch::zeros({batch, n_row, n_col}, sinogram.options());

    int threads = 256;
    int blocks = (int)((batch*n_ray + threads-1)/threads);

    back_project_2d_kernel<<<blocks, threads>>>(
        out_image.data_ptr<float>(),
        (int)batch, (int)n_row, (int)n_col,
        tvals.data_ptr<float>(),
        (int)n_intersections,
        sinogram.data_ptr<float>(),
        (int)n_ray,
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        M.data_ptr<float>(),
        b.data_ptr<float>()
    );

    return out_image;
}
