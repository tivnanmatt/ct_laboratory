from .core import RandomVariable

class RandomVariableFromLogProb(RandomVariable):
    def __init__(self, log_prob_function):
        super(RandomVariableFromLogProb, self).__init__()
        self.log_prob_function = log_prob_function

    def log_prob(self, *args, **kwargs):
        return self.log_prob_function(*args, **kwargs)