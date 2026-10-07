# File: setup.py

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ct_laboratory",
    # include all areas (tomography, optimization, random_variable,
    # sparse_eigen_preconditioner, bayesian_estimation, physics.*)
    packages=find_packages(include=["ct_laboratory", "ct_laboratory.*"]),
    package_data={"ct_laboratory.workflow": ["server_bootstrap.sh", "pod_check.py"]},
    entry_points={"console_scripts": ["ctlab=ct_laboratory.workflow.cli:main"]},
    ext_modules=[
        CUDAExtension(
            name="ct_laboratory._C",
            sources=[
                "src/bindings.cpp",
                "src/ct_projector_2d.cu",
                "src/ct_projector_3d.cu",
                "src/sf_projector_3d.cu",   # voxel-driven separable-footprint projector
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2", "--compiler-options", "-fPIC"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
