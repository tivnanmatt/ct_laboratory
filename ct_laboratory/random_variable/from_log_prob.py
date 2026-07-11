"""Random variable defined directly by a log-probability function."""
from .core import RandomVariable


class RandomVariableFromLogProb(RandomVariable):
    """Wrap any callable ``x -> scalar log probability``.

    Set ``normalized=False`` (default) when the callable is an unnormalized
    energy — sufficient for MAP estimation and scores, not for densities.
    """

    def __init__(self, log_prob_fn, normalized=False):
        super().__init__()
        self.log_prob_fn = log_prob_fn
        self.normalized = normalized

    def log_prob(self, x):
        return self.log_prob_fn(x)
