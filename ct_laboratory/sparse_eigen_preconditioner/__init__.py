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
