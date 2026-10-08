"""GPU eigensolvers for SparseEigenDecomposition (optional dependency: ``pip install ct_laboratory[gpu]`` = cupy-cuda12x).

``cupy_eigsh``: CuPy's thick-restart Lanczos (``cupyx.scipy.sparse.linalg.eigsh``) on the matrix-free Gram operator.
The Lanczos basis, its re-orthogonalization and the Rayleigh-Ritz step all stay on the GPU; the Gram matvec is the
same torch operator as the ARPACK path, handed between CuPy and torch zero-copy via DLPack.  Same interface and
outputs as ``eigsh`` (top-k eigenpairs of G = A^T A, descending).

Why: with ARPACK (scipy, host) every Lanczos step re-orthogonalizes against the whole basis on the CPU - measured
~100 ms per step at N = 131k on a shared 4090 host vs ~1.3 ms on the GPU - which dominated k = 1024 runs.
"""
from __future__ import annotations

import time

import torch

from .sparse_eigen_decomposition import SparseEigenDecomposition

__all__ = ["cupy_available"]


def cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def _solve_cupy_eigsh(self, gram, N, k, tol=1e-3, maxiter=5000, seed=42, ncv=None, verbose=False, use_tqdm=True, **_ignored):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, eigsh

    dev, dt = self.device_, self.dtype
    assert dev.type == "cuda", "cupy_eigsh needs the operator on a CUDA device"
    counter = {"n": 0}
    with cp.cuda.Device(dev.index or 0):
        def matvec(x):
            counter["n"] += 1
            # torch-owned copy of the input and an explicit sync on every device: CuPy and torch use different
            # streams, and the Gram may run on several GPUs (split projector); sharing buffers across them races
            xt = torch.from_dlpack(cp.ascontiguousarray(x.ravel().astype(cp.float32))).clone()
            cp.cuda.Device(dev.index or 0).synchronize()
            with torch.no_grad():
                gx = self._apply_gram(xt.to(dt)).detach().float().contiguous()
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            if verbose and (counter["n"] == 1 or counter["n"] % 250 == 0):
                print(f"  [cupy_eigsh] matvec {counter['n']}", flush=True)
            return cp.from_dlpack(gx).copy()

        op = LinearOperator((N, N), matvec=matvec, dtype=cp.float32)
        rs = cp.random.RandomState(int(seed) if seed is not None else None)
        v0 = rs.standard_normal(N, dtype=cp.float32)
        ncv = ncv if ncv is not None else min(N - 1, max(20, 2 * k + 1))
        t0 = time.time()
        w, V = eigsh(op, k=k, which="LM", ncv=ncv, v0=v0, tol=tol, maxiter=maxiter, return_eigenvectors=True)
        order = cp.argsort(w)[::-1]
        s2 = torch.from_dlpack(cp.ascontiguousarray(cp.clip(w[order], 0.0, None))).to(dt).clone()
        Vt = torch.from_dlpack(cp.ascontiguousarray(V[:, order])).to(dt).clone()
    if verbose:
        print(f"  [cupy_eigsh] done in {counter['n']} matvecs, {time.time() - t0:.1f} s", flush=True)
    self.last_solver_stats = dict(matvecs=counter["n"], ncv=ncv)
    return s2, Vt


SparseEigenDecomposition.register_method("cupy_eigsh", _solve_cupy_eigsh)
