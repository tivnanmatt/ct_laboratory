"""
Attenuation module for X-ray physics.

Provides tools for modeling X-ray attenuation through materials.
"""

from .mu_utils import get_mu
from .operators import (
    FixedAttenuator,
    UniformAluminumFilter,
    BasisAttenuator,
    ObjectAttenuator,
)

__all__ = [
    'get_mu',
    'FixedAttenuator',
    'UniformAluminumFilter',
    'BasisAttenuator',
    'ObjectAttenuator',
]
