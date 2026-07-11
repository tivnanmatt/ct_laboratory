"""Backward-compatibility shim — canonical location: ct_laboratory.tomography.standard_image_transform"""
import sys as _sys
from .tomography import standard_image_transform as _m
_sys.modules[__name__] = _m
