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
