
import torch

def inv_t_weighted_mse(x, x_hat, t):
    # NOTE: this computes mean(((y - y_hat) / t)^2)
    #       but the paper mentions mean((y - y_hat)^2 / t)
    assert isinstance(x, torch.Tensor)
    assert isinstance(x_hat, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    residual = x - x_hat
    # residual = residual / (t.unsqueeze(-1).unsqueeze(-1)).sqrt()
    return (residual ** 2).mean()