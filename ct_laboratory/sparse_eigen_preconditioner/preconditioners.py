"""Sparse-eigen preconditioners: optimization.Preconditioner + tomography.projector.

Given the top-k eigenpairs (s_i^2, v_i) of the Gram/Hessian operator
A^T A (v_i are image eigenvectors, p_i = A v_i are the corresponding
projection eigenvectors), the null-space-rescaled approximate inverse Hessian
is
      F = V (S^-2 - s_min^-2 I) V^T + s_min^-2 I,
and its symmetric square root P (P^2 = F) is the preconditioner
      P = V (S^-1 - s_min^-1 I) V^T + s_min^-1 I.
A single preconditioned-gradient step from zeros with lr=1 yields
      x_1 = P^2 (A^T y) = F (A^T y),
i.e. the sparse-eigen filtered back projection.  Both preconditioners below
implement the SAME operator P through two different computational pathways:
the IMAGE pathway stores V (N x k) and acts directly on the volume; the
PROJECTION (data) pathway stores p = A V (n_ray x k) and reaches the volume
only through the projector A, A^T.  k=1 collapses to s_max^-1 I, i.e. plain
scaled Landweber/steepest descent (the rescaled baseline).

Both classes subclass :class:`ct_laboratory.optimization.Preconditioner`, so
they satisfy the required forward + inverse contract and plug directly into
the ``bayesian_estimation`` estimators.
"""
import torch

from ..optimization import Preconditioner


class SparseEigenImagePreconditioner(Preconditioner):
    """Image-domain sparse-eigen preconditioner.

    P = V (S^-1 - s_min^-1 I) V^T + s_min^-1 I,   acting on a flattened volume.

    Parameters
    ----------
    s : (k,) tensor of singular values s_i = sqrt(eigenvalue_i) of A.
    v : (N, k) tensor of image eigenvectors (columns), N = n_x*n_y*n_z.
    """

    def __init__(self, s, v, eps=1e-12):
        super().__init__()
        self.register_buffer("v", v)
        s_min = s.min()
        self.register_buffer("diag_fwd", 1.0 / (s + eps) - 1.0 / (s_min + eps))
        self.register_buffer("diag_inv", (s - s_min))
        self.register_buffer("inv_s_min", 1.0 / (s_min + eps))
        self.register_buffer("s_min", s_min.clone())

    def forward(self, x):
        xf = x.reshape(-1)
        c = torch.mv(self.v.t(), xf) * self.diag_fwd
        return torch.mv(self.v, c) + self.inv_s_min * xf

    def inverse(self, x):
        """P^-1 = V (S - s_min I) V^T + s_min I."""
        xf = x.reshape(-1)
        c = torch.mv(self.v.t(), xf) * self.diag_inv
        return torch.mv(self.v, c) + self.s_min * xf


class SparseEigenProjectionPreconditioner(Preconditioner):
    """Projection-domain (data) sparse-eigen preconditioner.

    Mathematically identical to ``SparseEigenImagePreconditioner`` but realized
    through the projection eigenbasis p_i = A v_i and the projector, never
    touching the image eigenvectors V.  Using A^T p_i = s_i^2 v_i and
    p_i^T (A x) = s_i^2 v_i^T x,

        P(x) = A^T p_k D^-1 (S^-1 - s_min^-1 I) D^-1 p_k^T (A x) + s_min^-1 x,
        D = diag(s_i^2).

    Parameters
    ----------
    projector : object exposing ``forward_project`` and ``back_project``
        (e.g. a calibrated StaticCTProjector3D from ``tomography``).
    p : (n_ray, k) tensor of projection eigenvectors A v_i (may live on CPU).
    s2 : (k,) tensor of eigenvalues s_i^2 of A^T A.
    chunk : optional column block size for streaming p to the projector device;
        if None the whole basis is used in one matmul on its current device.
    """

    def __init__(self, projector, p, s2, eps=1e-12, chunk=None):
        super().__init__()
        self.projector = projector
        self.nx, self.ny, self.nz = projector.n_x, projector.n_y, projector.n_z
        self.dev = projector.M.device
        self.register_buffer("p", p)
        s2 = s2.to(self.dev)
        s = torch.sqrt(torch.clamp(s2, min=0.0))
        s_min = s.min()
        # weights folded onto the k-dim coefficient vector (everything in s^2 units):
        #   fwd:  D^-1 (S^-1 - s_min^-1) D^-1  = (s^-1 - s_min^-1) / s^4
        #   inv:  D^-1 (S   - s_min)   D^-1    = (s   - s_min)   / s^4
        s4 = torch.clamp(s2, min=eps).pow(2)
        self.register_buffer("w_fwd", (1.0 / (s + eps) - 1.0 / (s_min + eps)) / s4)
        self.register_buffer("w_inv", (s - s_min) / s4)
        self.register_buffer("inv_s_min", 1.0 / (s_min + eps))
        self.register_buffer("s_min", s_min.clone())
        self.chunk = chunk

    def _pk_T(self, r):
        """p_k^T r  ->  (k,) coefficient vector."""
        r = r.reshape(-1)
        if self.chunk is None:
            return torch.mv(self.p.to(self.dev).t(), r.to(self.p.to(self.dev).dtype))
        k = self.p.shape[1]
        out = torch.empty(k, device=self.dev)
        for a in range(0, k, self.chunk):
            b = min(k, a + self.chunk)
            out[a:b] = torch.mv(self.p[:, a:b].to(self.dev).t(), r)
        return out

    def _pk(self, c):
        """p_k c  ->  (n_ray,) projection vector."""
        if self.chunk is None:
            return torch.mv(self.p.to(self.dev), c.to(self.dev))
        k = self.p.shape[1]
        out = None
        for a in range(0, k, self.chunk):
            b = min(k, a + self.chunk)
            part = torch.mv(self.p[:, a:b].to(self.dev), c[a:b])
            out = part if out is None else out + part
        return out

    def _apply(self, x, w):
        xf = x.reshape(-1)
        ax = self.projector.forward_project(xf.view(self.nx, self.ny, self.nz))
        sino_shape = ax.shape
        c = self._pk_T(ax.reshape(-1)) * w
        proj = self._pk(c).view(sino_shape)
        return self.projector.back_project(proj).reshape(-1)

    def forward(self, x):
        xf = x.reshape(-1)
        return self._apply(xf, self.w_fwd) + self.inv_s_min * xf

    def inverse(self, x):
        xf = x.reshape(-1)
        return self._apply(xf, self.w_inv) + self.s_min * xf
