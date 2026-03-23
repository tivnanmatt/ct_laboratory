import torch
from torch import nn
import time
from tqdm import tqdm

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
        
        if debug:
            print(f"\n[MAP DEBUG] volume_pred.requires_grad={self.volume_pred.requires_grad}")
            print(f"[MAP DEBUG] volume_pred.shape={self.volume_pred.shape}")
            print(f"[MAP DEBUG] volume_pred.min/max={self.volume_pred.min():.6e}/{self.volume_pred.max():.6e}")
        
        log_likelihood = self.log_likelihood_fn(self.volume_pred)
        
        if debug:
            print(f"[MAP DEBUG] log_likelihood={log_likelihood:.6e}")
            print(f"[MAP DEBUG] log_likelihood.requires_grad={log_likelihood.requires_grad}")
        
        log_prior = self.log_prior_fn(self.volume_pred)
        
        if debug:
            print(f"[MAP DEBUG] log_prior={log_prior:.6e}")
        
        # Calculate Likelihood Grad Norm
        if log_likelihood.requires_grad:
            # OPTIMIZATION: Backward with retain_graph=True is expensive if we do it twice.
            # But the reconstructor does separate backprops for likelihood and prior to log norms.
            (-1.0 * log_likelihood).backward()
            if self.preconditioner is not None:
                grad_lik = self.volume_pred_preconditioned.grad.detach().clone()
            else:
                grad_lik = self.volume_pred.grad.detach().clone()
            
            if debug:
                print(f"[MAP DEBUG] After backward: grad_lik.norm={grad_lik.norm():.6e}")
                print(f"[MAP DEBUG] grad_lik.min/max={grad_lik.min():.6e}/{grad_lik.max():.6e}")
        else:
            if self.preconditioner is not None:
                grad_lik = torch.zeros_like(self.volume_pred_preconditioned)
            else:
                grad_lik = torch.zeros_like(self.volume_pred)
            if debug:
                print(f"[MAP DEBUG] log_likelihood has no grad, using zeros")

        # Calculate Prior Grad Norm (Base Score Function)
        self.optimizer.zero_grad()
        # Ensure volume_pred has grad reset if we're not using retain_graph
        if self.preconditioner is not None:
            self.volume_pred = self.preconditioner(self.volume_pred_preconditioned)

        if log_prior.requires_grad:
            (-1.0 * log_prior).backward()
            if self.preconditioner is not None:
                grad_prior = self.volume_pred_preconditioned.grad.detach().clone()
            else:
                grad_prior = self.volume_pred.grad.detach().clone()
            if debug:
                print(f"[MAP DEBUG] grad_prior.norm={grad_prior.norm():.6e}")
        else:
            if self.preconditioner is not None:
                grad_prior = torch.zeros_like(self.volume_pred_preconditioned)
            else:
                grad_prior = torch.zeros_like(self.volume_pred)
            if debug:
                print(f"[MAP DEBUG] log_prior has no grad, using zeros")

        # Restore total grad for optimizer
        if self.preconditioner is not None:
            if self.volume_pred_preconditioned.grad is None:
                self.volume_pred_preconditioned.grad = grad_lik + grad_prior
            else:
                self.volume_pred_preconditioned.grad.data = grad_lik + grad_prior
        else:
            if self.volume_pred.grad is None:
                self.volume_pred.grad = grad_lik + grad_prior
            else:
                self.volume_pred.grad.data = grad_lik + grad_prior
        
        grad_norm_lik = grad_lik.norm(2).item()
        grad_norm_prior = grad_prior.norm(2).item()

        log_posterior = log_likelihood + log_prior
        
        # Calculate final grad norm from optimizer's grad attribute
        if self.preconditioner is not None:
            gn_total = self.volume_pred_preconditioned.grad.detach().norm(2).item()
        else:
            gn_total = self.volume_pred.grad.detach().norm(2).item()
                
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        
        return log_likelihood.item(), log_prior.item(), log_posterior.item(), grad_norm_lik, grad_norm_prior, gn_total
    
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