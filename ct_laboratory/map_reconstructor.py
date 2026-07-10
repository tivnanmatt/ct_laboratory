import torch
from torch import nn
import time
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Sparse-eigen preconditioners for linear CT reconstruction.
#
# Given the top-k eigenpairs (s_i^2, v_i) of the Gram/Hessian operator
# A^T A (v_i are image eigenvectors, p_i = A v_i are the corresponding
# projection eigenvectors), the null-space-rescaled approximate inverse Hessian
# is
#       F = V (S^-2 - s_min^-2 I) V^T + s_min^-2 I,
# and its symmetric square root P (P^2 = F) is the preconditioner
#       P = V (S^-1 - s_min^-1 I) V^T + s_min^-1 I.
# A single preconditioned-gradient step from zeros with lr=1 yields
#       x_1 = P^2 (A^T y) = F (A^T y),
# i.e. the sparse-eigen filtered back projection.  Both preconditioners below
# implement the SAME operator P through two different computational pathways:
# the IMAGE pathway stores V (N x k) and acts directly on the volume; the
# PROJECTION (data) pathway stores p = A V (n_ray x k) and reaches the volume
# only through the projector A, A^T.  k=1 collapses to s_max^-1 I, i.e. plain
# scaled Landweber/steepest descent (the rescaled baseline).
# ---------------------------------------------------------------------------
class SparseEigenImagePreconditioner(nn.Module):
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


class SparseEigenProjectionPreconditioner(nn.Module):
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
        (e.g. a calibrated StaticCTProjector3D).
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


class MaximumAPosterioriReconstructor:
    """
    Class to perform MAP reconstruction given a log-likelihood and log-prior function.
    """
    def __init__(
        self,
        log_likelihood_fn,
        log_prior_fn,
        volume_init=None,
        preconditioner=None,
        inv_preconditioner=None,
        lr=None,
        warmup_iters=None
    ):
        self.log_likelihood_fn = log_likelihood_fn
        self.log_prior_fn = log_prior_fn
        self.preconditioner = preconditioner
        self.inv_preconditioner = inv_preconditioner
        
        
        if volume_init is not None:
            assert isinstance(volume_init, torch.Tensor)
            self.volume_pred = volume_init.clone().detach() 

        if lr is None:
            lr = 1.0

        if self.preconditioner is not None:
            self.volume_pred_preconditioned = self.inv_preconditioner(self.volume_pred)
            self.volume_pred_preconditioned = self.volume_pred_preconditioned.clone().detach().requires_grad_(True)
            self.optimizer = torch.optim.SGD([self.volume_pred_preconditioned], lr=lr)
        else:
            self.volume_pred = self.volume_pred.clone().detach().requires_grad_(True)
            self.optimizer = torch.optim.SGD([self.volume_pred], lr=lr)

        if warmup_iters is not None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda it: min(1.0, float(it) / warmup_iters)
            )
        else:
            self.scheduler = None

    def map_step(self, debug=False):
        self.optimizer.zero_grad()
        if self.preconditioner is not None:
            self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)
        
        log_likelihood = self.log_likelihood_fn(self.volume_pred)
        log_prior = self.log_prior_fn(self.volume_pred)
        log_posterior = log_likelihood + log_prior
        
        loss = -1.0 * log_posterior
        # Use retain_graph=True to allow multiple cycles if needed, 
        # but primarily we just need it to not break on the first iter.
        loss.backward()
        
        # Calculate Grad Norms for logging
        if self.preconditioner is not None:
            gn_total = self.volume_pred_preconditioned.grad.detach().norm(2).item()
            gn_lik = gn_total
            gn_prior = 0.0
        else:
            gn_total = self.volume_pred.grad.detach().norm(2).item()
            gn_lik = gn_total
            gn_prior = 0.0

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        # VERY IMPORTANT: Update volume_pred after step so the next iter sees correct values
        if self.preconditioner is not None:
            with torch.no_grad():
                self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)
        
        return log_likelihood.item(), log_prior.item(), log_posterior.item(), gn_lik, gn_prior, gn_total
    
    def map_reconstruction(self,
                           num_iters=100,
                           verbose=False,
                           use_tqdm=True,
                           randomize_likelihood_every=None,
                            print_every=1):
        t0 = time.time()
        print(f"Starting MAP reconstruction")
        
        range_it = range(1, num_iters + 1)
        if use_tqdm:
            range_it = tqdm(range_it, desc="MAP Reconstruction", leave=False)

        for it in range_it:
            
            if randomize_likelihood_every is not None:
                if it>0 and it % randomize_likelihood_every == 0:
                    self.log_likelihood_fn.randomize()

            log_likelihood, log_prior, log_posterior, gn_lik, gn_prior, gn_total = self.map_step()
            
            if use_tqdm:
                range_it.set_postfix({
                    "loss": f"{-1.0*log_posterior:.2e}",
                    "grad": f"{gn_total:.2e}",
                    "gn_lik": f"{gn_lik:.2e}",
                    "gn_score": f"{gn_prior:.2e}"
                })

            if verbose:
                if it % print_every == 0 or it == 1:
                    dt = time.time() - t0
                    if not use_tqdm:
                        print(f"Iter {it:04d}  likelihood={-1.0*log_likelihood:.4e}  prior={-1.0*log_prior:.4e} gn_opt={gn_total:.4e} ({dt:.2f} sec)")
                    t0 = time.time()
        
        if self.preconditioner is not None:
            with torch.no_grad():
                self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)
                
        return self.volume_pred
    
    
        

def map_reconstruction(
    vol_init_HU, 
    log_likelihood_fn, 
    log_prior_fn, 
    preconditioner=None, 
    inv_preconditioner=None,
    lr=100.0, 
    num_iters=200, 
    prior_weight=1.0, 
    warmup_iters=None, 
    verbose=False, 
    use_tqdm=True,
    print_every=1,
    history=None
):
    if preconditioner is not None:
        assert inv_preconditioner is not None
        vol_recon_HU_precond = inv_preconditioner(vol_init_HU)
        vol_recon_HU_precond = vol_recon_HU_precond.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([vol_recon_HU_precond], lr=lr)
        opt_var = vol_recon_HU_precond
    else:
        vol_recon_HU = vol_init_HU.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([vol_recon_HU], lr=lr)
        opt_var = vol_recon_HU


    if warmup_iters is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda it: min(1.0, float(it) / warmup_iters)
        )

    t0 = time.time()
    print(f"Starting MAP reconstruction  (prior λ = {prior_weight})")

    range_it = range(1, num_iters + 1)
    if use_tqdm:
        range_it = tqdm(range_it, desc=f"MAP Reconstruction (λ={prior_weight})", leave=False)

    for it in range_it:
        optimizer.zero_grad()
        if preconditioner is not None:
            vol_recon_HU = preconditioner(vol_recon_HU_precond)
        ll = log_likelihood_fn(vol_recon_HU)
        lp = log_prior_fn(vol_recon_HU)
        loss = -1.0 * (ll + prior_weight * lp)
        loss.backward()

        # Calculate grad norm for history/display
        if opt_var.grad is not None:
            gn = opt_var.grad.detach().norm(2).item()
        else:
            gn = 0.0

        if history is not None:
            if 'grad_norm' not in history:
                history['grad_norm'] = []
            if opt_var.grad is not None:
                if preconditioner is not None:
                    # Relationship: z = M x, where M is inv_preconditioner.
                    # L(x) = L(M^-1 z) => grad_z = (M^-1)^T grad_x
                    # grad_x = M^T grad_z.
                    # Since M is symmetric in our cases, grad_x = inv_preconditioner(opt_var.grad)
                    with torch.no_grad():
                        grad_x = inv_preconditioner(opt_var.grad)
                        history['grad_norm'].append(torch.norm(grad_x).item())
                else:
                    history['grad_norm'].append(torch.norm(opt_var.grad).item())
            else:
                history['grad_norm'].append(0.0)

        optimizer.step()

        if use_tqdm:
            range_it.set_postfix({
                "loss": f"{loss.item():.2e}",
                "grad": f"{gn:.2e}"
            })

        if warmup_iters is not None:
            scheduler.step()

        if verbose:
            if it % print_every == 0 or it == 1:
                dt = time.time() - t0
                if not use_tqdm:
                    print(f"Iter {it:04d}  data={ll.item():.4e}  prior={prior_weight * lp.item():.4e}  "
                        f"loss={loss.item():.4e}   ({dt:.2f} sec)")
                t0 = time.time()

    if preconditioner is not None:
        vol_recon_HU = preconditioner(vol_recon_HU_precond)

    return vol_recon_HU