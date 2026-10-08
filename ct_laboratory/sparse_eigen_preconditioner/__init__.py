"""Sparse-eigen preconditioner area.

The first area that COMBINES base areas: it depends on
``ct_laboratory.optimization`` (the ``Preconditioner`` contract) and on
``ct_laboratory.tomography`` projectors (the matrix-free Gram operator
G = A^T A whose leading eigenpairs are estimated here and turned into
preconditioners).
"""
from .sparse_eigen_decomposition import SparseEigenDecomposition
from .preconditioners import (SparseEigenImagePreconditioner,
                              SparseEigenProjectionPreconditioner)
from .weighted_preconditioner import WeightedSpectralSqrtPreconditioner
from .gpu_solvers import cupy_available   # registers the optional "cupy_eigsh" solver (needs cupy at call time only)
