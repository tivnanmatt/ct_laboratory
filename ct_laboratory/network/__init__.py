# __init__.py
#
# Merged from gmi (2026-07-11). Modules that depend on the optional `diffusers`
# package (DiffusersUnet2D, DiffusersUnet2D_Size28, MedMNISTDiffusion) are imported
# lazily so this area imports cleanly when `diffusers` is not installed. The native
# building blocks (SimpleCNN, DenseNet, LinearConv, LambdaLayer, ConfigurableUNet)
# have no such dependency and are imported eagerly.
from .simplecnn import SimpleCNN
from .densenet import DenseNet
from .linear_conv import LinearConv
from .lambda_layer import LambdaLayer
from .unet import ConfigurableUNet

__all__ = [
    "SimpleCNN",
    "DenseNet",
    "LinearConv",
    "LambdaLayer",
    "ConfigurableUNet",
    "DiffusersUnet2D",
    "DiffusersUnet2D_Size28",
    "MedMNISTDiffusion",
    "diffusion",
]


def __getattr__(name):
    # Lazy access to diffusers-dependent members (PEP 562).
    if name == "DiffusersUnet2D":
        from .diffusers_unet_2D import DiffusersUnet2D
        return DiffusersUnet2D
    if name == "DiffusersUnet2D_Size28":
        from .diffusers_unet_2D_28 import DiffusersUnet2D_Size28
        return DiffusersUnet2D_Size28
    if name == "MedMNISTDiffusion":
        from .diffusion import MedMNISTDiffusion
        return MedMNISTDiffusion
    if name == "diffusion":
        from . import diffusion
        return diffusion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
