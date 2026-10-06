"""Multi-device (multi-GPU) 3-D projector: one projector split into per-device sub-projectors.

The sinogram is partitioned into contiguous blocks (for the voxel projector: contiguous
VIEW ranges, i.e. source/panel pairs). Each block lives on its own device with its own
geometry and cache, so every device does 1/N of the work and holds 1/N of the cache:

    forward:  volume --copy--> every device --A_d--> y_d --copy--> output device, concatenated
    back:     y --split--> y_d on device d --A_d^T--> x_d --copy--> output device, summed

Kernels on different devices run concurrently because CUDA launches are asynchronous:
all devices are launched first and gathered afterwards. Consumer GPUs (RTX 4090/5090) have
no peer-to-peer access, so the copies go through host memory; that copy cost (one volume per
device per call) is what limits the speed-up at small sizes.

    A = split_voxel_projector(nx, ny, nz, M, b, views, valid=None,
                              devices=["cuda:0", "cuda:1", ...], cache="column")
    y = A.forward_project(x)        # x on A.output_device; y identical (up to float
    x = A.back_project(y)           # summation order) to the single-device VoxelProjector3D
    y = A(x)                        # autograd-aware, backward = back_project
"""
from __future__ import annotations

import torch

from .projector_3d_base import Projector3D
from .voxel_projector_3d_module import VoxelProjector3D

__all__ = ["MultiDeviceProjector3D", "split_voxel_projector"]


class _MultiDeviceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, proj):
        ctx.proj = proj
        return proj.forward_project(volume)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.proj.back_project(grad_output), None


class MultiDeviceProjector3D(Projector3D):
    """Concatenation of sub-projectors along the sinogram axis, each on its own device.

    Parameters
    ----------
    parts         : sub-projectors (any :class:`Projector3D`), same volume grid, in sinogram order
    output_device : device of inputs/outputs (default: the first part's device)
    """

    def __init__(self, parts, output_device=None):
        super().__init__()
        assert len(parts) > 0
        self.parts = torch.nn.ModuleList(parts)
        p0 = parts[0]
        self.n_x, self.n_y, self.n_z = p0.n_x, p0.n_y, p0.n_z
        for p in parts:
            assert (p.n_x, p.n_y, p.n_z) == (self.n_x, self.n_y, self.n_z), "all parts need the same grid"
        self.devices = [self._device_of(p) for p in parts]
        self.output_device = torch.device(output_device) if output_device is not None else self.devices[0]
        self._sizes = [int(p.n_ray) for p in parts]

    @staticmethod
    def _device_of(p):
        for t in list(p.buffers()) + list(p.parameters()):
            if t is not None:
                return t.device
        return torch.device("cpu")

    # ------------------------------------------------------------ properties
    @property
    def kind(self) -> str:
        return self.parts[0].kind

    @property
    def n_ray(self) -> int:
        return sum(self._sizes)

    @property
    def n_device(self) -> int:
        return len(self.parts)

    # ----------------------------------------------------------- projection
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Autograd-aware forward projection (backward = :meth:`back_project`)."""
        return _MultiDeviceFunction.apply(volume, self)

    def forward_project(self, volume: torch.Tensor) -> torch.Tensor:
        volume = volume.detach().contiguous()
        outs = []
        for p, d in zip(self.parts, self.devices):          # launch every device first ...
            with torch.cuda.device(d) if d.type == "cuda" else _null():
                outs.append(p.forward_project(volume.to(d, non_blocking=True)))
        return torch.cat([y.to(self.output_device, non_blocking=True) for y in outs], dim=-1)  # ... then gather

    def back_project(self, sinogram: torch.Tensor) -> torch.Tensor:
        sinogram = sinogram.detach().contiguous()
        outs = []
        for p, d, y in zip(self.parts, self.devices, torch.split(sinogram, self._sizes, dim=-1)):
            with torch.cuda.device(d) if d.type == "cuda" else _null():
                outs.append(p.back_project(y.to(d, non_blocking=True).contiguous()))
        acc = outs[0].to(self.output_device, non_blocking=True).clone()
        for x in outs[1:]:
            acc += x.to(self.output_device, non_blocking=True)
        return acc

    def extra_repr(self) -> str:
        return (f"kind={self.kind}, volume={self.volume_shape}, n_ray={self.n_ray}, "
                f"devices={[str(d) for d in self.devices]}, rays_per_device={self._sizes}")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def split_voxel_projector(n_x, n_y, n_z, M, b, views, valid=None, devices=None, output_device=None,
                          **kwargs) -> MultiDeviceProjector3D:
    """Build a :class:`VoxelProjector3D` split by contiguous view ranges over ``devices``.

    Views are divided so every device gets about the same number of detector pixels.
    ``kwargs`` (``backend``, ``cache``, ...) are passed to each :class:`VoxelProjector3D`.
    The sinogram order equals that of the single projector built from the same ``views``.
    """
    if devices is None:
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    devices = [torch.device(d) for d in devices]
    views = views.detach().to("cpu", torch.float32)
    npix = (views[:, 17].round() * views[:, 18].round()).to(torch.int64)
    cum = torch.cumsum(npix, 0)
    total = int(cum[-1])
    # view boundaries at ~equal pixel counts, at least one view per part
    n = min(len(devices), views.shape[0])
    cuts = [0] + [int(torch.searchsorted(cum, total * k / n).item()) + 1 for k in range(1, n)] + [views.shape[0]]
    for k in range(1, n):
        cuts[k] = min(max(cuts[k], cuts[k - 1] + 1), views.shape[0] - (n - k))
    off = torch.zeros(views.shape[0] + 1, dtype=torch.int64)
    off[1:] = cum
    valid_flat = None if valid is None else valid.reshape(-1)
    parts = []
    for k in range(n):
        v0, v1 = cuts[k], cuts[k + 1]
        val = None if valid_flat is None else valid_flat[int(off[v0]):int(off[v1])].to(devices[k])
        parts.append(VoxelProjector3D(n_x, n_y, n_z, M.to(devices[k]), b.to(devices[k]), views[v0:v1],
                                      valid=val, device=devices[k], **kwargs))
    return MultiDeviceProjector3D(parts, output_device=output_device or devices[0])
