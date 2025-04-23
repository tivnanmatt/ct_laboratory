// File: src/bindings.cpp

#include <torch/extension.h>

// Declare the three public functions from ct_projector_2d.cu
torch::Tensor compute_intersections_2d(
    int64_t n_row, int64_t n_col,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b
);

torch::Tensor forward_project_2d_cuda(
    torch::Tensor image,
    torch::Tensor tvals,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b
);

torch::Tensor back_project_2d_cuda(
    torch::Tensor sinogram,
    torch::Tensor tvals,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b,
    int64_t n_row,
    int64_t n_col
);

// Declarations of your 3D functions in ct_projector_3d.cu:
torch::Tensor compute_intersections_3d(
    int64_t n_x, int64_t n_y, int64_t n_z,
    torch::Tensor src, torch::Tensor dst,
    torch::Tensor M, torch::Tensor b
);

torch::Tensor forward_project_3d_cuda(
    torch::Tensor volume,
    torch::Tensor tvals,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b
);

torch::Tensor back_project_3d_cuda(
    torch::Tensor sinogram,
    torch::Tensor tvals,
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor M,
    torch::Tensor b,
    int64_t n_x,
    int64_t n_y,
    int64_t n_z
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_intersections_2d",
          &compute_intersections_2d,
          "Compute sorted intersection parameters (CUDA).");
    m.def("forward_project_2d_cuda",
          &forward_project_2d_cuda,
          "Forward projection using precomputed intersections (CUDA).");
    m.def("back_project_2d_cuda",
          &back_project_2d_cuda,
          "Back projection using precomputed intersections (CUDA).");
    m.def("compute_intersections_3d",
          &compute_intersections_3d,
          "Compute 3D intersections (CUDA).");
    m.def("forward_project_3d_cuda",
          &forward_project_3d_cuda,
          "Forward 3D projection (CUDA).");
    m.def("back_project_3d_cuda",
          &back_project_3d_cuda,
          "Back 3D projection (CUDA).");
}