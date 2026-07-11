from torch import optim

class LinearWarmupLRScheduler(optim.lr_scheduler.LambdaLR):
    '''
    Subclass of LambdaLR scheduler with lr_lambda fixed to a linear warmup.
    '''

    def __init__(
            self, 
            optimizer: optim.Optimizer, 
            warmup_steps: int, 
            **kwargs):
        
        # NOTE: this grows linearly to the provided constant lr
        lr_lambda = lambda epoch: min(1.0, (epoch + 1) / (warmup_steps + 1))
        
        super(LinearWarmupLRScheduler, self).__init__(optimizer, lr_lambda, **kwargs)