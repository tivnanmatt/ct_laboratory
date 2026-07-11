"""Optimization area — BASE AREA, no dependencies on other ct_laboratory areas.

Contains only generic optimization building blocks: the abstract
``Preconditioner`` base class (every preconditioner must implement BOTH
``forward`` and ``inverse``) and general loss functions. Nothing here knows
about projectors, physics, or probability models.
"""
from .preconditioner import Preconditioner, IdentityPreconditioner, DiagonalPreconditioner
from .loss_function import LossFunction, WeightedSumLoss, QuadraticLoss
