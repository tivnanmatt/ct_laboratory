"""
Attenuation operators for X-ray physics.

Provides two types of attenuation modules:
1. Fixed-length attenuators: Apply a pre-computed transmission (e.g., filters)
2. Variable-length attenuators: Compute transmission from basis integrals (e.g., objects)
"""

from .fixed_attenuator import FixedAttenuator, UniformAluminumFilter
from .basis_attenuator import BasisAttenuator, ObjectAttenuator

__all__ = [
    'FixedAttenuator',
    'UniformAluminumFilter',
    'BasisAttenuator',
    'ObjectAttenuator',
]
