// File: src/ct_projector_2d.cu
//
// 2025-04-30  – changes:
//   • kernels now expect **M_inv** (row-major, 4 floats) instead of M
//   • no on-the-fly inversion; helper only multiplies by the inverse
//   • midpoint voxel index uses IEEE “round-to-nearest-even” via __float2int_rn
//   • C++ wrapper signatures unchanged for Python, but the sixth argument is
//     now the inverse matrix

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>

// ────────────────────────────────────────────────────────────────
// helper: (x,y) → (row,col) with pre-computed inverse matrix
// layout: M_inv = [m11 m12 ; m21 m22] stored row-major
// ────────────────────────────────────────────────────────────────
__device__ __forceinline__
void apply_affine_inverse(
    float x, float y,
    const float* M_inv,   // [4]
    const float* b,       // [2]
    float& row, float& col)
{
    float xs = x - b[0];
    float ys = y - b[1];

    row = M_inv[0]*xs + M_inv[1]*ys;
    col = M_inv[2]*xs + M_inv[3]*ys;
}

// every kernel below:  argument renamed M_inv  (old code shown only
// where lines changed).

// ────────── kernel 1 – compute_intersections_2d_kernel ──────────
__global__ void compute_intersections_2d_kernel(
    int n_row, int n_col,
    const float* src_xy, const float* dst_xy, int n_ray,
    const float* M_inv, const float* b,
    float* t_out)
{
    int rid = blockIdx.x*blockDim.x + threadIdx.x;
    if (rid >= n_ray) return;

    float sx=src_xy[2*rid+0], sy=src_xy[2*rid+1];
    float dx=dst_xy[2*rid+0], dy=dst_xy[2*rid+1];

    float srx,sry, drx,dry;
    apply_affine_inverse(sx,sy, M_inv,b, srx,sry);
    apply_affine_inverse(dx,dy, M_inv,b, drx,dry);

    float dirx = drx - srx, diry = dry - sry;
    int n_int = (n_row+1)+(n_col+1);
    float* t_vals = &t_out[rid*n_int];
    int c=0;

    // row-planes
    for(int r=0;r<=n_row;++r){
        float plane=float(r)-0.5f;
        t_vals[c++] = (fabsf(dirx)<1e-12f)?INFINITY:
                      ([](float p,float s,float d){float t=(p-s)/d;return (t<0.f||t>1.f)?INFINITY:t;})(plane,srx,dirx);
    }
    // col-planes
    for(int c0=0;c0<=n_col;++c0){
        float plane=float(c0)-0.5f;
        t_vals[c++] = (fabsf(diry)<1e-12f)?INFINITY:
                      ([](float p,float s,float d){float t=(p-s)/d;return (t<0.f||t>1.f)?INFINITY:t;})(plane,sry,diry);
    }
    // bubble sort
    for(int i=0;i<n_int-1;++i)
        for(int j=0;j<n_int-1-i;++j)
            if(t_vals[j]>t_vals[j+1]){float tmp=t_vals[j];t_vals[j]=t_vals[j+1];t_vals[j+1]=tmp;}
}

// ────────── kernel 2 – forward_project_2d_kernel ──────────
__global__ void forward_project_2d_kernel(
    const float* image,int batch,int n_row,int n_col,
    const float* t_sorted,int n_int,
    const float* src_xy,const float* dst_xy,int n_ray,
    const float* M_inv,const float* b,
    float* out)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=batch*n_ray;
    if(gid>=total) return;

    int b_idx=gid/n_ray, r_idx=gid%n_ray;
    float sx=src_xy[2*r_idx+0], sy=src_xy[2*r_idx+1];
    float dx=dst_xy[2*r_idx+0], dy=dst_xy[2*r_idx+1];
    float vx=dx-sx, vy=dy-sy;

    const float* tvals=&t_sorted[r_idx*n_int];
    float acc=0.f;

    for(int i=0;i<n_int-1;++i){
        float t0=tvals[i], t1=tvals[i+1];
        if(isinf(t0)||isinf(t1)) continue;

        float x0=sx+t0*vx, y0=sy+t0*vy;
        float x1=sx+t1*vx, y1=sy+t1*vy;
        float seg_len=hypotf(x1-x0,y1-y0);

        float mx=0.5f*(x0+x1), my=0.5f*(y0+y1);
        float rf,cf; apply_affine_inverse(mx,my, M_inv,b, rf,cf);

        int ri=__float2int_rn(rf);
        int ci=__float2int_rn(cf);
        if(ri<0||ri>=n_row||ci<0||ci>=n_col) continue;

        size_t idx=(size_t)b_idx*n_row*n_col + (size_t)ri*n_col + ci;
        acc += image[idx]*seg_len;
    }
    out[b_idx*n_ray + r_idx]=acc;
}

// ────────── kernel 3 – back_project_2d_kernel ──────────
__global__ void back_project_2d_kernel(
    float* out_img,int batch,int n_row,int n_col,
    const float* t_sorted,int n_int,
    const float* sino,int n_ray,
    const float* src_xy,const float* dst_xy,
    const float* M_inv,const float* b)
{
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    int total=batch*n_ray;
    if(gid>=total) return;

    int b_idx=gid/n_ray, r_idx=gid%n_ray;
    float sx=src_xy[2*r_idx+0], sy=src_xy[2*r_idx+1];
    float dx=dst_xy[2*r_idx+0], dy=dst_xy[2*r_idx+1];
    float vx=dx-sx, vy=dy-sy;

    const float* tvals=&t_sorted[r_idx*n_int];
    float s_val=sino[b_idx*n_ray + r_idx];

    for(int i=0;i<n_int-1;++i){
        float t0=tvals[i], t1=tvals[i+1];
        if(isinf(t0)||isinf(t1)) continue;

        float x0=sx+t0*vx, y0=sy+t0*vy;
        float x1=sx+t1*vx, y1=sy+t1*vy;
        float seg_len=hypotf(x1-x0,y1-y0);

        float mx=0.5f*(x0+x1), my=0.5f*(y0+y1);
        float rf,cf; apply_affine_inverse(mx,my, M_inv,b, rf,cf);

        int ri=__float2int_rn(rf);
        int ci=__float2int_rn(cf);
        if(ri<0||ri>=n_row||ci<0||ci>=n_col) continue;

        float contrib=s_val*seg_len;
        size_t idx=(size_t)b_idx*n_row*n_col + (size_t)ri*n_col + ci;
        atomicAdd(&out_img[idx], contrib);
    }
}

// ────────── host wrappers – expect M_inv ──────────
torch::Tensor compute_intersections_2d(
    int64_t n_row,int64_t n_col,
    torch::Tensor src,torch::Tensor dst,
    torch::Tensor M_inv,torch::Tensor b)
{
    TORCH_CHECK(src.is_cuda()&&dst.is_cuda(),"src/dst must be CUDA");
    TORCH_CHECK(M_inv.is_cuda()&&b.is_cuda(),"M_inv/b must be CUDA");

    int64_t n_ray=src.size(0);
    int64_t n_int=(n_row+1)+(n_col+1);
    auto out=torch::empty({n_ray,n_int}, src.options().dtype(torch::kFloat32));

    int threads=256, blocks=(int)((n_ray+threads-1)/threads);
    compute_intersections_2d_kernel<<<blocks,threads>>>(
        (int)n_row,(int)n_col,
        src.data_ptr<float>(),dst.data_ptr<float>(),(int)n_ray,
        M_inv.data_ptr<float>(),b.data_ptr<float>(),
        out.data_ptr<float>());
    return out;
}

torch::Tensor forward_project_2d_cuda(
    torch::Tensor img,torch::Tensor tvals,
    torch::Tensor src,torch::Tensor dst,
    torch::Tensor M_inv,torch::Tensor b)
{
    TORCH_CHECK(img.is_cuda()&&tvals.is_cuda(),"img/tvals must be CUDA");

    int64_t batch=1,n_row,n_col;
    if(img.dim()==2){n_row=img.size(0);n_col=img.size(1);}
    else {batch=img.size(0);n_row=img.size(1);n_col=img.size(2);}
    int64_t n_ray=src.size(0), n_int=tvals.size(1);
    auto out=torch::zeros({batch,n_ray}, img.options());

    int threads=256, blocks=(int)((batch*n_ray+threads-1)/threads);
    forward_project_2d_kernel<<<blocks,threads>>>(
        img.data_ptr<float>(),(int)batch,(int)n_row,(int)n_col,
        tvals.data_ptr<float>(),(int)n_int,
        src.data_ptr<float>(),dst.data_ptr<float>(),(int)n_ray,
        M_inv.data_ptr<float>(),b.data_ptr<float>(),
        out.data_ptr<float>());
    return out;
}

torch::Tensor back_project_2d_cuda(
    torch::Tensor sino,torch::Tensor tvals,
    torch::Tensor src,torch::Tensor dst,
    torch::Tensor M_inv,torch::Tensor b,
    int64_t n_row,int64_t n_col)
{
    int64_t batch= (sino.dim()==2)?sino.size(0):1;
    int64_t n_ray = (sino.dim()==2)?sino.size(1):sino.size(0);
    int64_t n_int = tvals.size(1);
    auto out=torch::zeros({batch,n_row,n_col}, sino.options());

    int threads=256, blocks=(int)((batch*n_ray+threads-1)/threads);
    back_project_2d_kernel<<<blocks,threads>>>(
        out.data_ptr<float>(),(int)batch,(int)n_row,(int)n_col,
        tvals.data_ptr<float>(),(int)n_int,
        sino.data_ptr<float>(),(int)n_ray,
        src.data_ptr<float>(),dst.data_ptr<float>(),
        M_inv.data_ptr<float>(),b.data_ptr<float>());
    return out;
}
















// ──────────────────────────────────────────────────────────────────────
//               NEW: 2-D Siddon “on-the-fly” projection
//   • no pre-computed intersection list
//   • one thread = one (batch, ray) pair
//   • uses the same apply_affine_inverse helper
// ──────────────────────────────────────────────────────────────────────

// ────────── kernel  – forward_project_2d_on_the_fly ──────────
__global__ void forward_project_2d_on_the_fly_kernel(
    const float* img, int batch, int n_row, int n_col,
    const float* src_xy, const float* dst_xy, int n_ray,
    const float* M_inv, const float* b,
    float* out)
{
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (gid >= total) return;

    int b_idx = gid / n_ray;
    int r_idx = gid % n_ray;

    /* ---- physical ray -------------------------------------------------- */
    float sx = src_xy[2*r_idx+0], sy = src_xy[2*r_idx+1];
    float dx = dst_xy[2*r_idx+0], dy = dst_xy[2*r_idx+1];
    float vx = dx - sx,           vy = dy - sy;
    float ray_len = hypotf(vx, vy);
    if (ray_len < 1e-12f) { out[gid] = 0.f; return; }

    /* ---- map to row/col coordinates ------------------------------------ */
    float sr, sc, dr, dc;
    apply_affine_inverse(sx, sy, M_inv, b, sr, sc);
    apply_affine_inverse(dx, dy, M_inv, b, dr, dc);
    float vr = dr - sr,  vc = dc - sc;

    float inv_vr = (fabsf(vr) < 1e-12f) ? 1e32f : 1.f / vr;
    float inv_vc = (fabsf(vc) < 1e-12f) ? 1e32f : 1.f / vc;

    /* ---- entry / exit parameters --------------------------------------- */
    float t_r0 = (-0.5f              - sr) * inv_vr;
    float t_r1 = ((float)n_row-0.5f  - sr) * inv_vr;
    float t_c0 = (-0.5f              - sc) * inv_vc;
    float t_c1 = ((float)n_col-0.5f  - sc) * inv_vc;

    float t_enter = fmaxf(fminf(t_r0, t_r1), fminf(t_c0, t_c1));
    float t_exit  = fminf(fmaxf(t_r0, t_r1), fmaxf(t_c0, t_c1));
    if (t_exit <= t_enter) { out[gid] = 0.f; return; }

    t_enter = fmaxf(t_enter, 0.f);
    t_exit  = fminf(t_exit , 1.f);

    /* ---- initial voxel & stepping constants ---------------------------- */
    float r_enter = sr + t_enter*vr;
    float c_enter = sc + t_enter*vc;

    int ri = (vr >= 0.f) ? (int)floorf(r_enter) : (int)ceilf(r_enter) - 1;
    int ci = (vc >= 0.f) ? (int)floorf(c_enter) : (int)ceilf(c_enter) - 1;

    float t_next_r = (vr >= 0.f)
        ? t_enter + ((ri + 0.5f) - r_enter) * inv_vr
        : t_enter + ((ri - 0.5f) - r_enter) * inv_vr;
    float t_next_c = (vc >= 0.f)
        ? t_enter + ((ci + 0.5f) - c_enter) * inv_vc
        : t_enter + ((ci - 0.5f) - c_enter) * inv_vc;

    float dt_r = fabsf(inv_vr);
    float dt_c = fabsf(inv_vc);

    /* ---- Siddon march --------------------------------------------------- */
    float acc   = 0.f;
    float t_cur = t_enter;
    while (t_cur < t_exit)
    {
        bool step_row = (t_next_r < t_next_c);
        float t_hit   = step_row ? t_next_r : t_next_c;
        t_hit = fminf(t_hit, t_exit);           // clamp last step

        if (ri >= 0 && ri < n_row && ci >= 0 && ci < n_col)
        {
            float seg = (t_hit - t_cur) * ray_len;
            size_t idx = (size_t)b_idx*n_row*n_col + (size_t)ri*n_col + ci;
            acc += img[idx] * seg;
        }

        t_cur = t_hit;
        if (step_row) {
            t_next_r += dt_r;
            ri += (vr >= 0.f) ? 1 : -1;
        } else {
            t_next_c += dt_c;
            ci += (vc >= 0.f) ? 1 : -1;
        }
    }
    out[gid] = acc;
}

/* ────────── kernel  – back_project_2d_on_the_fly ────────── */
__global__ void back_project_2d_on_the_fly_kernel(
    float* img_out, int batch, int n_row, int n_col,
    const float* sino, int n_ray,
    const float* src_xy, const float* dst_xy,
    const float* M_inv, const float* b)
{
    int gid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_ray;
    if (gid >= total) return;

    int b_idx = gid / n_ray;
    int r_idx = gid % n_ray;

    float sx = src_xy[2*r_idx+0], sy = src_xy[2*r_idx+1];
    float dx = dst_xy[2*r_idx+0], dy = dst_xy[2*r_idx+1];
    float vx = dx - sx,           vy = dy - sy;
    float ray_len = hypotf(vx, vy);
    if (ray_len < 1e-12f) return;

    float sr, sc, dr, dc;
    apply_affine_inverse(sx, sy, M_inv, b, sr, sc);
    apply_affine_inverse(dx, dy, M_inv, b, dr, dc);
    float vr = dr - sr,  vc = dc - sc;

    float inv_vr = (fabsf(vr) < 1e-12f) ? 1e32f : 1.f / vr;
    float inv_vc = (fabsf(vc) < 1e-12f) ? 1e32f : 1.f / vc;

    float t_r0 = (-0.5f             - sr) * inv_vr;
    float t_r1 = ((float)n_row-0.5f - sr) * inv_vr;
    float t_c0 = (-0.5f             - sc) * inv_vc;
    float t_c1 = ((float)n_col-0.5f - sc) * inv_vc;

    float t_enter = fmaxf(fminf(t_r0,t_r1), fminf(t_c0,t_c1));
    float t_exit  = fminf(fmaxf(t_r0,t_r1), fmaxf(t_c0,t_c1));
    if (t_exit <= t_enter) return;

    t_enter = fmaxf(t_enter, 0.f);
    t_exit  = fminf(t_exit , 1.f);

    float r_enter = sr + t_enter*vr;
    float c_enter = sc + t_enter*vc;

    int ri = (vr >= 0.f) ? (int)floorf(r_enter) : (int)ceilf(r_enter)-1;
    int ci = (vc >= 0.f) ? (int)floorf(c_enter) : (int)ceilf(c_enter)-1;

    float t_next_r = (vr >= 0.f)
        ? t_enter + ((ri + 0.5f) - r_enter) * inv_vr
        : t_enter + ((ri - 0.5f) - r_enter) * inv_vr;
    float t_next_c = (vc >= 0.f)
        ? t_enter + ((ci + 0.5f) - c_enter) * inv_vc
        : t_enter + ((ci - 0.5f) - c_enter) * inv_vc;

    float dt_r = fabsf(inv_vr);
    float dt_c = fabsf(inv_vc);
    float proj_val = sino[gid];

    float t_cur = t_enter;
    while (t_cur < t_exit)
    {
        bool step_row = (t_next_r < t_next_c);
        float t_hit   = step_row ? t_next_r : t_next_c;
        t_hit = fminf(t_hit, t_exit);

        if (ri >= 0 && ri < n_row && ci >= 0 && ci < n_col)
        {
            float seg = (t_hit - t_cur) * ray_len;
            size_t idx = (size_t)b_idx*n_row*n_col + (size_t)ri*n_col + ci;
            atomicAdd(&img_out[idx], proj_val * seg);
        }

        t_cur = t_hit;
        if (step_row) {
            t_next_r += dt_r;
            ri += (vr >= 0.f) ? 1 : -1;
        } else {
            t_next_c += dt_c;
            ci += (vc >= 0.f) ? 1 : -1;
        }
    }
}

/* ────────── host wrappers – on-the-fly versions ────────── */
torch::Tensor forward_project_2d_on_the_fly_cuda(
    torch::Tensor img,
    torch::Tensor src, torch::Tensor dst,
    torch::Tensor M_inv, torch::Tensor b)
{
    TORCH_CHECK(img.is_cuda() && src.is_cuda() && dst.is_cuda(),
                "inputs must be CUDA tensors");

    int64_t batch = (img.dim()==2) ? 1 : img.size(0);
    int64_t n_row = (img.dim()==2) ? img.size(0) : img.size(1);
    int64_t n_col = (img.dim()==2) ? img.size(1) : img.size(2);
    int64_t n_ray = src.size(0);

    auto out = torch::zeros({batch, n_ray}, img.options());

    int threads = 256;
    int blocks  = (int)((batch*n_ray + threads - 1) / threads);
    forward_project_2d_on_the_fly_kernel<<<blocks, threads>>>(
        img.data_ptr<float>(), (int)batch, (int)n_row, (int)n_col,
        src.data_ptr<float>(), dst.data_ptr<float>(), (int)n_ray,
        M_inv.data_ptr<float>(), b.data_ptr<float>(),
        out.data_ptr<float>());

    return out;
}

torch::Tensor back_project_2d_on_the_fly_cuda(
    torch::Tensor sino,
    torch::Tensor src, torch::Tensor dst,
    torch::Tensor M_inv, torch::Tensor b,
    int64_t n_row, int64_t n_col)
{
    TORCH_CHECK(sino.is_cuda() && src.is_cuda() && dst.is_cuda(),
                "inputs must be CUDA tensors");

    int64_t batch = (sino.dim()==1) ? 1 : sino.size(0);
    int64_t n_ray = (sino.dim()==1) ? sino.size(0) : sino.size(1);

    auto out = torch::zeros({batch, n_row, n_col}, sino.options());

    int threads = 256;
    int blocks  = (int)((batch*n_ray + threads - 1) / threads);
    back_project_2d_on_the_fly_kernel<<<blocks, threads>>>(
        out.data_ptr<float>(), (int)batch, (int)n_row, (int)n_col,
        sino.data_ptr<float>(), (int)n_ray,
        src.data_ptr<float>(), dst.data_ptr<float>(),
        M_inv.data_ptr<float>(), b.data_ptr<float>());

    return out;
}
