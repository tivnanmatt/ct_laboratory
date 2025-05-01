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

/* ────────────────────────────── 3-D  (on-the-fly) ─────────────────────────── */
torch::Tensor forward_project_3d_on_the_fly_cuda(torch::Tensor,
                                                 torch::Tensor,torch::Tensor,
                                                 torch::Tensor,torch::Tensor);

torch::Tensor back_project_3d_on_the_fly_cuda(torch::Tensor,
                                              torch::Tensor,torch::Tensor,
                                              torch::Tensor,torch::Tensor,
                                              int64_t,int64_t,int64_t);

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

    /* -------- 3-D, Siddon on-the-fly ------------------------------------- */
    m.def("forward_project_3d_on_the_fly_cuda",
          &forward_project_3d_on_the_fly_cuda,
          "Forward projection (3-D, Siddon on-the-fly, CUDA)");
    m.def("back_project_3d_on_the_fly_cuda",
          &back_project_3d_on_the_fly_cuda,
          "Back projection (3-D, Siddon on-the-fly, CUDA)");
}
