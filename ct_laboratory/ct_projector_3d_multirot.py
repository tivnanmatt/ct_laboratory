"""Backward-compatibility shim — canonical location: ct_laboratory.tomography.ct_projector_3d_multirot"""
import sys as _sys
from .tomography import ct_projector_3d_multirot as _m
_sys.modules[__name__] = _m
