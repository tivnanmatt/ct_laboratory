"""CT system area: tomography.projector + physics.xray_system integration.

``XraySystem`` (physics/xray) is deliberately projector-free: it maps basis
LINE INTEGRALS ``x_basis [n_rays, n_materials]`` to spectral detector
statistics. ``CTSystem`` is the layer that closes the loop with geometry: it
holds a tomography projector A (any object exposing ``forward_project`` /
``back_project``) and composes

    basis volumes [n_materials, nx, ny, nz]
        --A-->  x_basis [n_rays, n_materials]
        --XraySystem-->  detector signals y [n_rays, n_channels]

so the full spectral measurement model of a scan is one differentiable
module.
"""
import torch
from torch import nn


class CTSystem(nn.Module):
    """End-to-end spectral CT forward model: projector + X-ray system.

    Parameters
    ----------
    projector : tomography projector
        Object exposing ``forward_project(volume) -> sinogram`` and
        ``back_project(sinogram) -> volume`` (e.g. ``CTProjector3DModule`` or
        any of its geometry subclasses).
    xray_system : ct_laboratory.physics.xray.xray_system.XraySystem
        Spectral measurement model operating on basis line integrals.
    """

    def __init__(self, projector, xray_system):
        super().__init__()
        for required in ("forward_project", "back_project"):
            if not hasattr(projector, required):
                raise TypeError(f"projector must expose {required!r}; got "
                                f"{type(projector).__name__}")
        self.projector = projector
        self.xray_system = xray_system

    @property
    def n_materials(self):
        return self.xray_system.basis_materials.n_materials

    def project_basis(self, basis_volumes):
        """Basis volumes -> basis line integrals.

        basis_volumes : [n_materials, nx, ny, nz] (or a single volume for a
        one-material basis). Returns x_basis [n_rays, n_materials].
        """
        if basis_volumes.dim() == 3:
            basis_volumes = basis_volumes.unsqueeze(0)
        cols = [self.projector.forward_project(vol).reshape(-1)
                for vol in basis_volumes]
        return torch.stack(cols, dim=1)

    def back_project_basis(self, x_basis):
        """Adjoint of :meth:`project_basis`:
        [n_rays, n_materials] -> [n_materials, nx, ny, nz]."""
        vols = [self.projector.back_project(x_basis[:, m])
                for m in range(x_basis.shape[1])]
        return torch.stack(vols, dim=0)

    def forward_stats(self, basis_volumes, **kwargs):
        """Mean and variance of the detector signal for basis volumes."""
        x_basis = self.project_basis(basis_volumes)
        return self.xray_system.forward_stats(x_basis, **kwargs)

    def forward(self, basis_volumes, **kwargs):
        """Mean detector signal y for basis volumes."""
        x_basis = self.project_basis(basis_volumes)
        return self.xray_system(x_basis, **kwargs)

    def log_prob(self, y_obs, basis_volumes, **kwargs):
        """log p(y_obs | basis volumes) under the spectral noise model."""
        x_basis = self.project_basis(basis_volumes)
        return self.xray_system.log_prob(y_obs, x_basis, **kwargs)

    def sample(self, basis_volumes, **kwargs):
        """Draw a noisy measurement for basis volumes."""
        x_basis = self.project_basis(basis_volumes)
        return self.xray_system.sample(x_basis, **kwargs)

    def decompose_basis(self, y_obs, **kwargs):
        """Ray-domain material decomposition (delegates to the X-ray system)."""
        return self.xray_system.decompose_basis(y_obs, **kwargs)
