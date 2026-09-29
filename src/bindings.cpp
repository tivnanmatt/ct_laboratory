// File: src/bindings.cpp
#include <torch/extension.h>

/* ────────────────────────────── 2-D  (pre-computed) ───────────────────────── */
torch::Tensor compute_intersections_2d(int64_t,int64_t,
                                       torch::Tensor,torch::Tensor,
                                       torch::Tensor,torch::Tensor);

torch::Tensor forward_project_2d_cuda(torch::Tensor,torch::Tensor,
                                      torch::Tensor,torch::Tensor,
                                      torch::Tensor,torch::Tensor);

torch::Tensor back_project_2d_cuda(torch::Tensor,torch::Tensor,
                                   torch::Tensor,torch::Tensor,
                                   torch::Tensor,torch::Tensor,
                                   int64_t,int64_t);

/* ────────────────────────────── 2-D  (on-the-fly) ─────────────────────────── */
torch::Tensor forward_project_2d_on_the_fly_cuda(torch::Tensor,
                                                 torch::Tensor,torch::Tensor,
                                                 torch::Tensor,torch::Tensor);

torch::Tensor back_project_2d_on_the_fly_cuda(torch::Tensor,
                                              torch::Tensor,torch::Tensor,
                                              torch::Tensor,torch::Tensor,
                                              int64_t,int64_t);

/* ────────────────────────────── 3-D  (pre-computed) ───────────────────────── */
torch::Tensor compute_intersections_3d(int64_t,int64_t,int64_t,
                                       torch::Tensor,torch::Tensor,
                                       torch::Tensor,torch::Tensor);

torch::Tensor forward_project_3d_cuda(torch::Tensor,torch::Tensor,
                                      torch::Tensor,torch::Tensor,
                                      torch::Tensor,torch::Tensor);

torch::Tensor back_project_3d_cuda(torch::Tensor,torch::Tensor,
                                   torch::Tensor,torch::Tensor,
                                   torch::Tensor,torch::Tensor,
                                   int64_t,int64_t,int64_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compress_tvals_3d_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, double);

torch::Tensor forward_project_3d_compressed_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor back_project_3d_compressed_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);

/* ────────────────────────────── 3-D  (on-the-fly) ─────────────────────────── */
torch::Tensor forward_project_3d_on_the_fly_cuda(torch::Tensor,
                                                 torch::Tensor,torch::Tensor,
                                                 torch::Tensor,torch::Tensor);

torch::Tensor back_project_3d_on_the_fly_cuda(torch::Tensor,
                                              torch::Tensor,torch::Tensor,
                                              torch::Tensor,torch::Tensor,
                                              int64_t,int64_t,int64_t);

/* ─────────────────── 3-D  VOXEL-DRIVEN, separable footprints ──────────────── */
torch::Tensor sf_forward_project_3d_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor sf_back_project_3d_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sf_precompute_footprints_3d_cuda(
    int64_t, int64_t, int64_t,
    torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor);

torch::Tensor sf_forward_project_3d_csr_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);

torch::Tensor sf_back_project_3d_csr_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, int64_t);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sf_precompute_columns_3d_cuda(
    int64_t nx, int64_t ny, int64_t nz, torch::Tensor M, torch::Tensor b, torch::Tensor views, torch::Tensor view_off, torch::Tensor valid);
torch::Tensor sf_forward_project_3d_col_cuda(torch::Tensor volume, torch::Tensor M, torch::Tensor b, torch::Tensor views, torch::Tensor view_off, torch::Tensor valid,
    torch::Tensor col_ptr, torch::Tensor col_view, torch::Tensor col_trap);
torch::Tensor sf_back_project_3d_col_cuda(torch::Tensor sino, torch::Tensor M, torch::Tensor b, torch::Tensor views, torch::Tensor view_off, torch::Tensor valid,
    torch::Tensor col_ptr, torch::Tensor col_view, torch::Tensor col_trap, int64_t nx, int64_t ny, int64_t nz);


/* ========================================================================== */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /* -------- 2-D, pre-computed ------------------------------------------ */
    m.def("compute_intersections_2d", &compute_intersections_2d,
          "Compute sorted intersection parameters (2-D, CUDA)");
    m.def("forward_project_2d_cuda",  &forward_project_2d_cuda,
          "Forward projection (2-D, pre-computed intersections, CUDA)");
    m.def("back_project_2d_cuda",     &back_project_2d_cuda,
          "Back projection (2-D, pre-computed intersections, CUDA)");

    /* -------- 2-D, Siddon on-the-fly ------------------------------------- */
    m.def("forward_project_2d_on_the_fly_cuda",
          &forward_project_2d_on_the_fly_cuda,
          "Forward projection (2-D, Siddon on-the-fly, CUDA)");
    m.def("back_project_2d_on_the_fly_cuda",
          &back_project_2d_on_the_fly_cuda,
          "Back projection (2-D, Siddon on-the-fly, CUDA)");

    /* -------- 3-D, pre-computed ------------------------------------------ */
    m.def("compute_intersections_3d", &compute_intersections_3d,
          "Compute 3-D intersections (CUDA)");
    m.def("forward_project_3d_cuda",  &forward_project_3d_cuda,
          "Forward projection (3-D, pre-computed intersections, CUDA)");
    m.def("back_project_3d_cuda",     &back_project_3d_cuda,
          "Back projection (3-D, pre-computed intersections, CUDA)");

    m.def("compress_tvals_3d_cuda", &compress_tvals_3d_cuda,
          "Compress 3-D tvals to uint16 deltas (CUDA)");
    m.def("forward_project_3d_compressed_cuda", &forward_project_3d_compressed_cuda,
          "Forward projection (3-D, compressed uint16 intersections, CUDA)");
    m.def("back_project_3d_compressed_cuda", &back_project_3d_compressed_cuda,
          "Back projection (3-D, compressed uint16 intersections, CUDA)");

    /* -------- 3-D, Siddon on-the-fly ------------------------------------- */
    m.def("forward_project_3d_on_the_fly_cuda",
          &forward_project_3d_on_the_fly_cuda,
          "Forward projection (3-D, Siddon on-the-fly, CUDA)");
    m.def("back_project_3d_on_the_fly_cuda",
          &back_project_3d_on_the_fly_cuda,
          "Back projection (3-D, Siddon on-the-fly, CUDA)");

    /* -------- 3-D, voxel-driven separable footprints --------------------- */
    m.def("sf_forward_project_3d_cuda", &sf_forward_project_3d_cuda,
          "Forward projection (3-D, voxel-driven separable footprints, on-the-fly, CUDA)");
    m.def("sf_back_project_3d_cuda", &sf_back_project_3d_cuda,
          "Back projection (3-D, voxel-driven separable footprints, on-the-fly, CUDA)");
    m.def("sf_precompute_footprints_3d_cuda", &sf_precompute_footprints_3d_cuda,
          "Precompute per-voxel (pixel index, weight) footprint lists as CSR (CUDA)");
    m.def("sf_forward_project_3d_csr_cuda", &sf_forward_project_3d_csr_cuda,
          "Forward projection (3-D, precomputed voxel footprints, CUDA)");
    m.def("sf_back_project_3d_csr_cuda", &sf_back_project_3d_csr_cuda,
          "Back projection (3-D, precomputed voxel footprints, CUDA)");
    m.def("sf_precompute_columns_3d_cuda", &sf_precompute_columns_3d_cuda,
          "SF: per-column culled view list + stored transaxial trapezoids");
    m.def("sf_forward_project_3d_col_cuda", &sf_forward_project_3d_col_cuda,
          "SF forward projection using the column cache");
    m.def("sf_back_project_3d_col_cuda", &sf_back_project_3d_col_cuda,
          "SF back projection using the column cache");
}
