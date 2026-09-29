"""Abstract base class for 3-D projectors.

Two concrete families derive from it:

* :class:`RayProjector3D`   (alias of :class:`CTProjector3DModule`) — RAY-driven.
  One thread per ray; the system matrix is stored as per-ray voxel-intersection
  parameters (``tvals``, Siddon).  Forward projection is a gather, back
  projection is a scatter (atomics on the volume).

* :class:`VoxelProjector3D` — VOXEL-driven, separable footprints (SF-TR).
  One thread per voxel; the system matrix is stored (or recomputed) as per-voxel
  lists of (detector pixel index, weight).  Back projection is a gather (no
  atomics), forward projection is a scatter (atomics on the sinogram).

Both expose the same interface so reconstruction code can be written once:

    proj.forward_project(volume)  -> sinogram      (autograd-aware via proj(volume))
    proj.back_project(sinogram)   -> volume        (the exact adjoint)
    proj.n_x, proj.n_y, proj.n_z, proj.volume_shape, proj.n_ray
"""
from __future__ import annotations

import torch

__all__ = ["Projector3D"]


class Projector3D(torch.nn.Module):
    """Common interface for ray-driven and voxel-driven 3-D projectors."""

    #: volume dimensions (set by subclasses)
    n_x: int
    n_y: int
    n_z: int

    # ------------------------------------------------------------------ API
    def forward_project(self, volume: torch.Tensor) -> torch.Tensor:
        """``[X,Y,Z]`` or ``[B,X,Y,Z]`` volume -> sinogram ``[n_ray]`` or ``[B,n_ray]``."""
        raise NotImplementedError

    def back_project(self, sinogram: torch.Tensor) -> torch.Tensor:
        """Adjoint: sinogram ``[n_ray]`` or ``[B,n_ray]`` -> volume."""
        raise NotImplementedError

    def forward(self, volume: torch.Tensor) -> torch.Tensor:  # nn.Module call
        return self.forward_project(volume)

    # ------------------------------------------------------------- helpers
    @property
    def volume_shape(self) -> tuple[int, int, int]:
        return (self.n_x, self.n_y, self.n_z)

    @property
    def n_voxel(self) -> int:
        return self.n_x * self.n_y * self.n_z

    @property
    def n_ray(self) -> int:
        """Number of sinogram entries (rays / detector pixels)."""
        raise NotImplementedError

    @property
    def kind(self) -> str:
        """'ray' or 'voxel'."""
        raise NotImplementedError

    def extra_repr(self) -> str:
        try:
            return f"kind={self.kind}, volume={self.volume_shape}, n_ray={self.n_ray}"
        except NotImplementedError:
            return f"volume={self.volume_shape}"
