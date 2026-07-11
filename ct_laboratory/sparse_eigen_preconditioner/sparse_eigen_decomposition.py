import torch
from torch import nn
import time
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Sparse eigen-decomposition of the CT Gram / Hessian operator A^T A.
#
# The sparse-eigen preconditioners in ``map_reconstructor.py`` (Image / Projection
# flavours) *consume* the top-k eigenpairs (s_i^2, v_i) of A^T A but do not compute
# them.  This module fills that gap: it estimates the leading eigenpairs of the
# (typically huge, never-materialized) symmetric PSD operator
#
#       G = A^T A,     G v_i = s_i^2 v_i,     p_i = A v_i,
#
# using only matrix-vector products with the projector (matrix-free).  Results are
# stored as ``nn.Module`` buffers so a decomposition can be saved once (expensive)
# and reloaded cheaply via the normal PyTorch ``state_dict`` machinery.
#
# Typical use
# -----------
#   dec = SparseEigenDecomposition(projector, k=64)
#   dec.compute_weights(n_iters=30, verbose=True)     # the expensive part
#   dec.save("eig_k64.pt")                             # or torch.save(dec.state_dict())
#   ...
#   dec = SparseEigenDecomposition(projector, k=64)
#   dec.load("eig_k64.pt")                             # cheap reload
#   P    = dec.to_image_preconditioner()               # hand to MAP reconstructor
#
# Extending
# ---------
# The eigensolver is dispatched by name.  A new solver is a function
# ``fn(self, gram, N, k, **kw) -> (s2, V)`` registered with
# ``SparseEigenDecomposition.register_method("my_solver", fn)`` and selected via
# ``compute_weights(method="my_solver")``.  The default "subspace" (block power +
# Rayleigh-Ritz) is dependency-free and robust for the leading spectrum.
# ---------------------------------------------------------------------------
class SparseEigenDecomposition(nn.Module):
    """Matrix-free leading eigen-decomposition of the CT Gram operator ``A^T A``.

    Parameters
    ----------
    operator : projector or callable
        Either a CT projector exposing ``forward_project`` / ``back_project`` and
        integer shape attributes ``n_x, n_y, n_z`` (e.g. ``StaticCTProjector3D``),
        or a plain callable implementing the forward map ``A`` (volume -> sinogram),
        in which case ``volume_shape`` must be given.  If ``gram`` is supplied
        instead, ``operator`` may be ``None``.
    k : int
        Number of leading eigenpairs to estimate.
    volume_shape : tuple[int, ...], optional
        Required only when ``operator`` is a bare forward callable.
    gram : callable, optional
        Direct symmetric operator ``x -> A^T A x`` acting on a *flattened* volume.
        If given it overrides the projector-derived Gram; projection eigenvectors
        ``p_i = A v_i`` are then unavailable.
    dtype, device : torch dtype / device
        Working precision and placement of the stored basis.  Defaults follow the
        projector (``projector.M``) when available, else float32 / CPU.

    Buffers (populated by :meth:`compute_weights`, restored by load)
    ---------------------------------------------------------------
    s   : (k,)      singular values s_i = sqrt(eigenvalue_i), descending.
    v   : (N, k)    image-domain eigenvectors (columns), N = prod(volume_shape).
    p   : (n_ray,k) projection-domain eigenvectors A v_i (only if requested and a
                    projector is available), else a 0-length tensor.
    """

    # solver registry: name -> callable(self, gram, N, k, **kw) -> (s2, V)
    _METHODS = {}

    def __init__(self, operator=None, k=32, volume_shape=None, gram=None,
                 dtype=None, device=None):
        super().__init__()
        self.k = int(k)

        self._projector = None
        self._forward = None
        self._gram_user = gram

        # ---- resolve operator into forward / gram / shape / device -----------
        if gram is None and operator is None:
            raise ValueError("provide either an `operator` (projector or forward "
                             "callable) or a `gram` callable")

        if hasattr(operator, "forward_project") and hasattr(operator, "back_project"):
            self._projector = operator
            self._forward = operator.forward_project
            volume_shape = (operator.n_x, operator.n_y, operator.n_z)
            if device is None and hasattr(operator, "M"):
                device = operator.M.device
        elif callable(operator):
            if volume_shape is None and gram is None:
                raise ValueError("`volume_shape` is required when `operator` is a "
                                 "bare forward callable")
            self._forward = operator

        self.volume_shape = tuple(volume_shape) if volume_shape is not None else None
        self.dtype = dtype or torch.float32
        self.device_ = device or torch.device("cpu")

        # ---- placeholder buffers (resized on compute / load) -----------------
        self.register_buffer("s", torch.zeros(0, dtype=self.dtype))
        self.register_buffer("v", torch.zeros(0, 0, dtype=self.dtype))
        self.register_buffer("p", torch.zeros(0, dtype=self.dtype))

    # -- operator plumbing ----------------------------------------------------
    def _apply_gram(self, x):
        """G x for a single flattened vector x (shape (N,))."""
        if self._gram_user is not None:
            return self._gram_user(x).reshape(-1)
        # The CUDA projector reads raw data_ptr() and assumes a contiguous
        # volume; a strided view (e.g. a column slice Q[:, j]) is silently
        # misread. Force contiguity before every forward/back projection.
        vol = x.contiguous().reshape(self.volume_shape)
        return self._projector.back_project(
            self._projector.forward_project(vol)).reshape(-1)

    def _gram_matmul(self, Q, use_tqdm=False, desc="G @ Q"):
        """Apply G to every column of Q (N x k) -> (N x k), column by column."""
        cols = range(Q.shape[1])
        if use_tqdm:
            cols = tqdm(cols, desc=desc, leave=False)
        out = torch.empty_like(Q)
        for j in cols:
            out[:, j] = self._apply_gram(Q[:, j])
        return out

    @property
    def N(self):
        if self.volume_shape is not None:
            n = 1
            for d in self.volume_shape:
                n *= d
            return n
        return self.v.shape[0] if self.v.numel() else None

    # -- solvers --------------------------------------------------------------
    def _solve_subspace(self, gram, N, k, n_iters=30, tol=1e-6, seed=None,
                        oversample=0, verbose=False, use_tqdm=True):
        """Block power (subspace) iteration + Rayleigh-Ritz for the top-k eigenpairs.

        Iterates Q <- orth(G Q) to capture the dominant invariant subspace, then a
        small Rayleigh-Ritz step yields the eigen-pairs restricted to that subspace.
        ``oversample`` extra probe vectors improve accuracy of the k-th pair.
        """
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device_).manual_seed(int(seed))
        m = min(N, k + max(0, int(oversample)))
        Q = torch.randn(N, m, dtype=self.dtype, device=self.device_, generator=gen)
        Q, _ = torch.linalg.qr(Q)

        prev = None
        it_range = range(int(n_iters))
        if use_tqdm:
            it_range = tqdm(it_range, desc="subspace iter", leave=False)
        for it in it_range:
            Z = gram(Q, use_tqdm=False)
            Q, _ = torch.linalg.qr(Z)
            # cheap convergence proxy: Ritz values on the current subspace
            T = Q.t() @ gram(Q, use_tqdm=False)
            T = 0.5 * (T + T.t())
            ritz = torch.linalg.eigvalsh(T)
            if prev is not None:
                rel = (ritz - prev).abs().max() / (prev.abs().max() + 1e-30)
                if verbose:
                    tqdm.write(f"  [subspace] iter {it:03d}  max Ritz={ritz.max():.4e}  "
                               f"rel-change={rel:.2e}")
                if rel < tol:
                    break
            prev = ritz

        # final Rayleigh-Ritz projection
        GQ = gram(Q, use_tqdm=False)
        T = Q.t() @ GQ
        T = 0.5 * (T + T.t())
        evals, W = torch.linalg.eigh(T)         # ascending
        idx = torch.argsort(evals, descending=True)[:k]
        s2 = torch.clamp(evals[idx], min=0.0)
        V = Q @ W[:, idx]
        return s2, V

    def _solve_eigsh(self, gram, N, k, tol=1e-3, maxiter=5000, seed=42,
                     ncv=None, verbose=False, use_tqdm=True, **_ignored):
        """Leading top-k eigenpairs of ``G = A^T A`` via scipy ``eigsh(which='LM')``.

        This is the implicitly-restarted Lanczos path (ARPACK) that produced all
        prior sparse-eig paper results.  The Gram matvec runs on the projector's
        device (GPU); scipy drives the Krylov iteration on the host with float32
        numpy.  Typical cost is ~3*k matvecs.  More accurate for the leading
        spectrum than the dependency-free "subspace" solver, at the price of a
        scipy dependency (imported lazily here).
        """
        import numpy as np
        from scipy.sparse.linalg import LinearOperator, eigsh

        vshape = self.volume_shape
        dev, dt = self.device_, self.dtype
        counter = {"n": 0}
        expected = max(1, int(3.05 * k))

        def matvec(x):
            counter["n"] += 1
            xt = torch.as_tensor(np.asarray(x), dtype=dt, device=dev).reshape(vshape)
            with torch.no_grad():
                gx = self._apply_gram(xt.reshape(-1))
            if verbose and (counter["n"] == 1 or counter["n"] % 25 == 0):
                print(f"  [eigsh] matvec {counter['n']}/~{expected}", flush=True)
            return gx.detach().to("cpu", torch.float32).numpy()

        op = LinearOperator((N, N), matvec=matvec, dtype=np.float32)
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        v0 = rng.standard_normal(N).astype(np.float32)
        ncv = ncv if ncv is not None else min(N - 1, max(20, 2 * k + 1))
        s2_np, V_np = eigsh(op, k=k, which="LM", ncv=ncv, v0=v0,
                            tol=tol, maxiter=maxiter)
        order = np.argsort(s2_np)[::-1]
        s2 = torch.as_tensor(np.clip(s2_np[order], 0.0, None).copy(), dtype=dt, device=dev)
        V = torch.as_tensor(V_np[:, order].copy(), dtype=dt, device=dev)
        if verbose:
            print(f"  [eigsh] done in {counter['n']} matvecs", flush=True)
        return s2, V

    # -- public API -----------------------------------------------------------
    def compute_weights(self, k=None, method="subspace", compute_projection_basis=None,
                        verbose=False, use_tqdm=True, **kwargs):
        """Estimate the leading eigenpairs of ``A^T A`` and populate the buffers.

        Parameters
        ----------
        k : int, optional              override the number of eigenpairs.
        method : str                   registered solver name (default "subspace").
        compute_projection_basis : bool, optional
            Also store ``p_i = A v_i`` (needs a projector).  Defaults to True when a
            projector is available, else False.
        **kwargs                       forwarded to the chosen solver (e.g. n_iters,
                                       tol, seed, oversample).

        Returns
        -------
        self  (buffers ``s``, ``v`` and optionally ``p`` are filled in place).
        """
        if k is not None:
            self.k = int(k)
        k = self.k
        N = self.N
        if N is None:
            raise RuntimeError("cannot infer problem size N; give volume_shape or gram")

        solver = self._METHODS.get(method)
        if solver is None:
            raise KeyError(f"unknown method '{method}'; registered: "
                           f"{sorted(self._METHODS)}")

        def gram(Q, use_tqdm=False):
            return self._gram_matmul(Q, use_tqdm=use_tqdm)

        t0 = time.time()
        if verbose:
            print(f"[SparseEigenDecomposition] method={method}  N={N}  k={k}")
        s2, V = solver(self, gram, N, k, verbose=verbose, use_tqdm=use_tqdm, **kwargs)

        s = torch.sqrt(torch.clamp(s2, min=0.0)).to(self.dtype)
        V = V.to(self.dtype)
        self.register_buffer("s", s.contiguous())
        self.register_buffer("v", V.contiguous())

        if compute_projection_basis is None:
            compute_projection_basis = self._forward is not None and self._projector is not None
        if compute_projection_basis:
            if self._forward is None:
                raise RuntimeError("projection basis requested but no forward operator")
            p_cols = []
            cols = range(V.shape[1])
            if use_tqdm:
                cols = tqdm(cols, desc="p = A v", leave=False)
            for j in cols:
                # contiguous(): strided column view is misread by CUDA projector
                proj = self._forward(V[:, j].contiguous().reshape(self.volume_shape))
                p_cols.append(proj.reshape(-1))
            self.register_buffer("p", torch.stack(p_cols, dim=1).contiguous())
        else:
            self.register_buffer("p", torch.zeros(0, dtype=self.dtype))

        if verbose:
            print(f"[SparseEigenDecomposition] done in {time.time()-t0:.2f}s  "
                  f"s in [{s.min():.3e}, {s.max():.3e}]  cond~={(s.max()/torch.clamp(s.min(),min=1e-30)):.2e}")
        return self

    @property
    def eigenvalues(self):
        """Eigenvalues s_i^2 of A^T A (descending)."""
        return self.s ** 2

    def to_image_preconditioner(self, eps=1e-12):
        """Build a ``SparseEigenImagePreconditioner`` from the stored basis."""
        from .preconditioners import SparseEigenImagePreconditioner
        self._require_computed()
        return SparseEigenImagePreconditioner(self.s, self.v, eps=eps)

    def to_projection_preconditioner(self, projector=None, eps=1e-12, chunk=None):
        """Build a ``SparseEigenProjectionPreconditioner`` from the stored basis."""
        from .preconditioners import SparseEigenProjectionPreconditioner
        self._require_computed()
        if self.p.numel() == 0:
            raise RuntimeError("projection basis not available; recompute with "
                               "compute_projection_basis=True")
        projector = projector or self._projector
        if projector is None:
            raise ValueError("a projector is required to build a projection preconditioner")
        return SparseEigenProjectionPreconditioner(
            projector, self.p, self.eigenvalues, eps=eps, chunk=chunk)

    def _require_computed(self):
        if self.v.numel() == 0:
            raise RuntimeError("no eigen-basis; call compute_weights() or load() first")

    # -- save / load (PyTorch state_dict based) -------------------------------
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # buffers are dynamically sized; resize placeholders to match the checkpoint
        for name in ("s", "v", "p"):
            key = prefix + name
            if key in state_dict:
                self.register_buffer(name, torch.empty_like(state_dict[key]))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def save(self, path):
        """Save the decomposition (state_dict + shape metadata) to ``path``."""
        torch.save({
            "state_dict": self.state_dict(),
            "k": self.k,
            "volume_shape": self.volume_shape,
        }, path)
        return path

    def load(self, path, map_location=None):
        """Load a decomposition saved by :meth:`save` (or a raw state_dict)."""
        ckpt = torch.load(path, map_location=map_location or self.device_)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.k = ckpt.get("k", self.k)
            if ckpt.get("volume_shape") is not None:
                self.volume_shape = tuple(ckpt["volume_shape"])
            state = ckpt["state_dict"]
        else:
            state = ckpt
        self.load_state_dict(state)
        return self

    @classmethod
    def register_method(cls, name, fn):
        """Register a new eigensolver ``fn(self, gram, N, k, **kw) -> (s2, V)``."""
        cls._METHODS[name] = fn
        return fn

    def extra_repr(self):
        computed = "computed" if self.v.numel() else "empty"
        return f"k={self.k}, volume_shape={self.volume_shape}, basis={computed}"


# built-in solvers
SparseEigenDecomposition.register_method(
    "subspace", SparseEigenDecomposition._solve_subspace)
SparseEigenDecomposition.register_method(
    "eigsh", SparseEigenDecomposition._solve_eigsh)
