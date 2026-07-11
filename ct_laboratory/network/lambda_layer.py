from torch import nn

class LambdaLayer(nn.Module):
            def __init__(self, lambda_fn):
                super(LambdaLayer, self).__init__()
                self.lambda_fn = lambda_fn
            
            def forward(self, x):
                return self.lambda_fn(x)