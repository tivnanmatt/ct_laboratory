"""
Materials module for X-ray physics.

This module provides tools for working with materials in X-ray CT simulations,
including:
- Material composition and density definitions
- Attenuation coefficient calculations using xraydb
- Basis material decomposition (e.g., photoelectric/Compton)
- Material property lookups and interpolation
"""

from .material_database import MaterialDatabase, PHANTOM_MATERIALS
from .basis_materials import BasisMaterials
from .mu_utils import get_mu, get_mu_photoelectric, get_mu_compton
from .standard_bases import PECSBasis, WaterBasis, SoftTissueBasis

__all__ = [
    'MaterialDatabase',
    'PHANTOM_MATERIALS',
    'BasisMaterials',
    'get_mu',
    'get_mu_photoelectric',
    'get_mu_compton',
    'PECSBasis',
    'WaterBasis',
    'SoftTissueBasis',
]
