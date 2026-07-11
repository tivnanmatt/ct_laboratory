"""Random-variable area — BASE AREA, no dependencies on other ct_laboratory areas.

Random variables and distributions are the same thing; we call them random
variables. ``RandomVariable`` (unconditional) is the prior concept;
``ConditionalRandomVariable`` is the likelihood concept — no additional
prior/likelihood aliases exist beyond these two. torch.distributions
interoperate through ``FromTorchDistribution`` /
``ConditionalFromTorchDistribution`` and ``RandomVariable.from_torch``.
"""
from .core import RandomVariable, ConditionalRandomVariable
from .from_torch_distribution import (FromTorchDistribution,
                                      ConditionalFromTorchDistribution)
from .from_log_prob import RandomVariableFromLogProb
from .gaussian import (DiagonalGaussianRandomVariable,
                       ConditionalGaussianRandomVariable,
                       LinearGaussianRandomVariable,
                       AdditiveWhiteGaussianNoise)
from .poisson import (PoissonRandomVariable,
                      ConditionalPoissonRandomVariable,
                      BeerLambertPoissonRandomVariable)
from .image_energies import (QuadraticSmoothnessRandomVariable,
                             TotalVariationRandomVariable)

# Distribution-style aliases for those who prefer the other name.
Distribution = RandomVariable
ConditionalDistribution = ConditionalRandomVariable
