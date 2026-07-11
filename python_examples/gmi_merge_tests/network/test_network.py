"""Tests for the merged gmi `network` area under `ct_laboratory.network`.

Covers the native (diffusers-free) building blocks and the new native
`ConfigurableUNet` (2D and 3D). The diffusers-backed 2D U-Nets are exercised
separately and skipped when `diffusers` is not installed.

All tests run on CPU. Regression note: SimpleCNN's `dim=3` path was broken in
gmi (called an undefined `get_activation` and hard-coded a `Conv2d` output
layer); `test_simplecnn_3d_*` guards the fix.
"""
import pytest
import torch

from ct_laboratory.network import (
    SimpleCNN,
    DenseNet,
    LinearConv,
    LambdaLayer,
    ConfigurableUNet,
)


# --------------------------------------------------------------------------- #
# SimpleCNN
# --------------------------------------------------------------------------- #
def test_simplecnn_2d_forward_shape():
    m = SimpleCNN(1, 1, [4, 8], "relu", dim=2)
    y = m(torch.randn(2, 1, 16, 16))
    assert y.shape == (2, 1, 16, 16)


def test_simplecnn_3d_forward_shape():
    # Regression: previously raised AttributeError (undefined get_activation).
    m = SimpleCNN(1, 2, [4, 8], "relu", dim=3)
    y = m(torch.randn(2, 1, 8, 8, 8))
    assert y.shape == (2, 2, 8, 8, 8)


def test_simplecnn_3d_uses_conv3d_output_layer():
    # Regression: output layer must be Conv3d for dim=3 (was hard-coded Conv2d).
    m = SimpleCNN(1, 3, [4], "relu", dim=3)
    last_conv = [mod for mod in m.model if isinstance(mod, torch.nn.modules.conv._ConvNd)][-1]
    assert isinstance(last_conv, torch.nn.Conv3d)
    assert last_conv.out_channels == 3


def test_simplecnn_invalid_dim_raises():
    with pytest.raises(ValueError):
        SimpleCNN(1, 1, [4], "relu", dim=4)


def test_simplecnn_backward():
    m = SimpleCNN(1, 1, [4], "relu", dim=2)
    y = m(torch.randn(1, 1, 8, 8))
    y.sum().backward()
    assert any(p.grad is not None for p in m.parameters())


# --------------------------------------------------------------------------- #
# Other native blocks
# --------------------------------------------------------------------------- #
def test_densenet_forward_shape():
    m = DenseNet(input_shape=[1, 8, 8], output_shape=[1, 8, 8], hidden_channels_list=[32, 16])
    y = m(torch.randn(4, 1, 8, 8))
    assert y.shape == (4, 1, 8, 8)


def test_linearconv_forward_shape():
    m = LinearConv(in_channels=1, out_channels=2, kernel_size=5)
    y = m(torch.randn(3, 1, 12, 12))
    assert y.shape == (3, 2, 12, 12)


def test_lambda_layer_identity():
    m = LambdaLayer(lambda x: x * 2)
    x = torch.randn(2, 3)
    assert torch.allclose(m(x), x * 2)


# --------------------------------------------------------------------------- #
# ConfigurableUNet (native, 2D + 3D)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("channel_mults,num_res_blocks", [((1, 2), 1), ((1, 2, 4), 2)])
def test_unet_2d_forward_shape(channel_mults, num_res_blocks):
    m = ConfigurableUNet(
        dim=2, in_channels=1, out_channels=1, base_channels=8,
        channel_mults=channel_mults, num_res_blocks=num_res_blocks,
    )
    y = m(torch.randn(2, 1, 32, 32))
    assert y.shape == (2, 1, 32, 32)


def test_unet_2d_channel_change():
    m = ConfigurableUNet(dim=2, in_channels=3, out_channels=1, base_channels=8, channel_mults=(1, 2))
    y = m(torch.randn(2, 3, 16, 16))
    assert y.shape == (2, 1, 16, 16)


def test_unet_2d_time_embedding():
    m = ConfigurableUNet(dim=2, in_channels=2, out_channels=1, base_channels=8, time_embedding=True)
    y = m(torch.randn(2, 2, 32, 32), torch.tensor([3, 7]))
    assert y.shape == (2, 1, 32, 32)


def test_unet_3d_forward_shape():
    m = ConfigurableUNet(dim=3, in_channels=1, out_channels=1, base_channels=8, channel_mults=(1, 2), num_res_blocks=1)
    y = m(torch.randn(1, 1, 16, 16, 16))
    assert y.shape == (1, 1, 16, 16, 16)


def test_unet_3d_time_embedding():
    m = ConfigurableUNet(dim=3, in_channels=1, out_channels=2, base_channels=8, channel_mults=(1, 2, 4), time_embedding=True)
    y = m(torch.randn(1, 1, 16, 16, 16), torch.tensor([5]))
    assert y.shape == (1, 2, 16, 16, 16)


def test_unet_3d_backward():
    m = ConfigurableUNet(dim=3, base_channels=8, channel_mults=(1, 2))
    y = m(torch.randn(1, 1, 16, 16, 16))
    y.sum().backward()
    assert any(p.grad is not None for p in m.parameters())


def test_unet_time_embedding_default_timestep():
    # time_embedding=True but t omitted -> defaults to zeros, still runs.
    m = ConfigurableUNet(dim=2, base_channels=8, channel_mults=(1, 2), time_embedding=True)
    y = m(torch.randn(2, 1, 16, 16))
    assert y.shape == (2, 1, 16, 16)


def test_unet_invalid_dim_raises():
    with pytest.raises(ValueError):
        ConfigurableUNet(dim=1)


def test_unet_nondivisible_groups_ok():
    # base_channels not divisible by default norm_num_groups=8 -> must not crash.
    m = ConfigurableUNet(dim=2, base_channels=6, channel_mults=(1, 2), norm_num_groups=8)
    y = m(torch.randn(1, 1, 16, 16))
    assert y.shape == (1, 1, 16, 16)


# --------------------------------------------------------------------------- #
# Diffusers-backed 2D U-Nets (ported from gmi; require `diffusers`)
# --------------------------------------------------------------------------- #
def test_diffusers_unet_2d_size28_forward():
    pytest.importorskip("diffusers")
    from ct_laboratory.network import DiffusersUnet2D_Size28
    m = DiffusersUnet2D_Size28(in_channels=1, out_channels=1)
    y = m(torch.randn(2, 1, 28, 28), torch.zeros(2, dtype=torch.long))
    assert y.shape == (2, 1, 28, 28)


def test_diffusers_unet_2d_forward():
    pytest.importorskip("diffusers")
    from ct_laboratory.network import DiffusersUnet2D
    m = DiffusersUnet2D(image_size=32, unet_in_channels=1, unet_base_channels=8, unet_out_channels=1)
    y = m(torch.randn(1, 1, 32, 32), torch.tensor([5]))
    assert y.shape == (1, 1, 32, 32)


def test_medmnist_diffusion_construct():
    # Regression: MedMNISTDiffusion used DenseNet without importing it.
    pytest.importorskip("diffusers")
    from ct_laboratory.network import MedMNISTDiffusion
    m = MedMNISTDiffusion(image_shape=[1, 28, 28])
    assert isinstance(m, torch.nn.Module)
