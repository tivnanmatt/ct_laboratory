"""Physics areas.

* ``physics.xray`` — spectral X-ray measurement physics. Operates purely on
  basis LINE INTEGRALS (``x_basis [n_rays, n_materials]``); it has NO
  dependency on projectors or any other ct_laboratory area.
* ``physics.ct_system`` — integrates a ``tomography`` projector with an
  ``XraySystem`` into one differentiable spectral scan model.
"""
from . import xray
from . import ct_system
from .ct_system import CTSystem
