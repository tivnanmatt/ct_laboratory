"""Bayesian estimation area.

Depends on ``random_variable`` (likelihood = ConditionalRandomVariable,
prior = unconditional RandomVariable) and ``optimization`` (the
Preconditioner contract). Does not depend on tomography directly — projector
operators enter only through the random variables that close over them.
"""
from .map_estimator import MaximumAPosterioriEstimator
