"""
Dataset-free smoke tests for the merged diffusion stack
(ct_laboratory.diffusion + its generative deps: random_variable_gmi, samplers).

The gmi `datasets` area was intentionally NOT merged, so these tests avoid any
data loading: a trivial backbone and synthetic tensors exercise the training
loss closure and the reverse-process sampler on CPU. This proves the merged
diffusion/SDE/random-variable stack imports and computes; full training/
sampling on real data needs the datasets + network areas (not merged).
"""
import torch
import pytest

from ct_laboratory.diffusion import DiffusionModel
from ct_laboratory.sde import VariancePreservingSDE, StandardWienerSDE
from ct_laboratory.random_variable_gmi import UniformRandomVariable
from ct_laboratory.samplers import Sampler


class DummyBackbone(torch.nn.Module):
    """Trivial x_0 estimator: returns x_t unchanged."""
    def forward(self, x_t, t):
        return x_t


class Loss3(torch.nn.Module):
    """3-arg loss matching DiffusionModel's (x0, x0_pred, t) call convention."""
    def forward(self, x0, x0pred, t):
        return torch.mean((x0 - x0pred) ** 2)


@pytest.fixture
def model():
    return DiffusionModel(DummyBackbone(), training_loss_fn=Loss3())


def test_generative_deps_import():
    assert issubclass(UniformRandomVariable, Sampler)


def test_training_loss_is_scalar(model):
    x0 = torch.randn(2, 1, 8, 8)
    loss = model(x0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_reverse_process_shape_and_finite(model):
    xt = torch.randn(2, 1, 8, 8)
    timesteps = torch.linspace(1.0, 1e-3, 5)
    out = model.sample_reverse_process(xt, timesteps, sampler='euler')
    assert out.shape == xt.shape
    assert torch.isfinite(out).all()


def test_reverse_process_heun(model):
    xt = torch.randn(1, 1, 8, 8)
    timesteps = torch.linspace(1.0, 1e-3, 4)
    out = model.sample_reverse_process(xt, timesteps, sampler='heun')
    assert out.shape == xt.shape


@pytest.mark.parametrize("sde_cls", [VariancePreservingSDE, StandardWienerSDE])
def test_custom_forward_sde(sde_cls):
    m = DiffusionModel(DummyBackbone(), forward_SDE=sde_cls(), training_loss_fn=Loss3())
    loss = m(torch.randn(2, 1, 8, 8))
    assert torch.isfinite(loss)
