"""Backward-compatibility shim — canonical location: ct_laboratory.tomography.ct_projector_2d_torch"""
import sys as _sys
from .tomography import ct_projector_2d_torch as _m
_sys.modules[__name__] = _m
