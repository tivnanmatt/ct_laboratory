# File: setup.py

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="ct_laboratory",
    packages=["ct_laboratory"],
    ext_modules=[
        CUDAExtension(
            name="ct_laboratory._C",
            sources=[
                "src/bindings.cpp",
                "src/ct_projector_2d.cu",
                "src/ct_projector_3d.cu"
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2", "--compiler-options", "-fPIC"]
            }
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
