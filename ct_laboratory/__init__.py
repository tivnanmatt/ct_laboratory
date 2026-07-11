# ---------------------------------------------------------------------------
# ct_laboratory — area layout (dependencies flow strictly downward):
#
#   tomography          BASE: all projectors (2D/3D, fan/cone/StaticCT,
#                       multirot, subsets, grid transforms)
#   optimization        BASE: Preconditioner contract (forward AND inverse
#                       required) + general loss functions; no projector deps
#   random_variable     BASE: RandomVariable / ConditionalRandomVariable on
#                       top of torch.distributions; priors ARE unconditional
#                       random variables, likelihoods ARE conditional ones
#   sparse_eigen_preconditioner
#                       optimization.Preconditioner + tomography.projector:
#                       matrix-free Gram eigenpairs and the preconditioners
#                       built from them
#   bayesian_estimation random_variable + optimization: MAP estimation
#   physics.xray        spectral X-ray measurement physics (line-integral
#                       domain only; no projector dependencies)
#   physics.ct_system   tomography.projector + physics.xray_system
# ---------------------------------------------------------------------------
from . import tomography
from . import optimization
from . import random_variable
from . import sparse_eigen_preconditioner
from . import bayesian_estimation
from . import physics

# ---------------------------------------------------------------------------
# Backward-compatible flat namespace (legacy imports keep working).
# ---------------------------------------------------------------------------
from .tomography import *
from .diagonal_gaussian_prior import *
from .nonlinear_poisson_likelihood import *
from .linear_gaussian_likelihood import *
from .map_reconstructor import *
from .sparse_eigen_preconditioner import (SparseEigenDecomposition,
                                          SparseEigenImagePreconditioner,
                                          SparseEigenProjectionPreconditioner)
from .bayesian_diffusion_posterior_sampling import *
from .quadratic_smoothness_prior import *
from .total_variance_prior import *
from .bayesian_denoiser_prior import *

from .optimization import Preconditioner, IdentityPreconditioner
from .random_variable import RandomVariable, ConditionalRandomVariable
from .bayesian_estimation import MaximumAPosterioriEstimator
from .physics import CTSystem
