"""Backward-compatibility shim — canonical location: ct_laboratory.sparse_eigen_preconditioner.sparse_eigen_decomposition"""
import sys as _sys
from .sparse_eigen_preconditioner import sparse_eigen_decomposition as _m
_sys.modules[__name__] = _m
