"""Tests for the merged gmi `loss_function` and `lr_scheduler` areas.

Both areas are small and self-contained (no external deps beyond torch).
CPU-only.
"""
import torch

from ct_laboratory.loss_function import inv_t_weighted_mse
from ct_laboratory.lr_scheduler import LinearWarmupLRScheduler


# --------------------------------------------------------------------------- #
# loss_function.inv_t_weighted_mse
# --------------------------------------------------------------------------- #
def test_inv_t_weighted_mse_finite_scalar():
    pred = torch.randn(4, 1, 8, 8)
    tgt = torch.randn(4, 1, 8, 8)
    t = torch.rand(4) + 0.1
    loss = inv_t_weighted_mse(pred, tgt, t)
    assert torch.isfinite(loss).all()


def test_inv_t_weighted_mse_zero_on_match():
    x = torch.randn(3, 1, 8, 8)
    t = torch.rand(3) + 0.1
    loss = inv_t_weighted_mse(x, x, t)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_inv_t_weighted_mse_currently_ignores_t():
    # KNOWN BEHAVIOR (flagged for Matt): despite the name, the 1/t weighting is
    # commented out in the source, so this is plain MSE and `t` has no effect.
    # This guards the *present* behavior; update if the weighting is restored.
    pred = torch.ones(1, 1, 4, 4)
    tgt = torch.zeros(1, 1, 4, 4)
    loss_small_t = inv_t_weighted_mse(pred, tgt, torch.tensor([0.1]))
    loss_large_t = inv_t_weighted_mse(pred, tgt, torch.tensor([0.9]))
    assert torch.allclose(loss_small_t, loss_large_t)


# --------------------------------------------------------------------------- #
# lr_scheduler.LinearWarmupLRScheduler
# --------------------------------------------------------------------------- #
def _make_opt(lr=0.1):
    p = [torch.nn.Parameter(torch.randn(2))]
    return torch.optim.SGD(p, lr=lr)


def test_linear_warmup_ramps_up():
    opt = _make_opt(lr=0.1)
    sched = LinearWarmupLRScheduler(opt, warmup_steps=5)
    lrs = []
    for _ in range(7):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    # LR should increase across warmup.
    assert lrs[0] < lrs[4]
    # And plateau at (or below) the base LR afterwards.
    assert lrs[-1] <= 0.1 + 1e-9


def test_linear_warmup_reaches_base_lr():
    base = 0.05
    opt = _make_opt(lr=base)
    sched = LinearWarmupLRScheduler(opt, warmup_steps=4)
    for _ in range(4):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - base) < 1e-6
