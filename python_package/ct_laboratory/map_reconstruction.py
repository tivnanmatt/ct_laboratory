import torch
import time

def map_reconstruction(
    vol_init_HU, log_likelihood_fn, log_prior_fn, 
    preconditioner=None, inv_preconditioner=None,
    lr=100.0, num_iters=200, prior_weight=1.0, 
    warmup_iters=None, verbose=False, print_every=1
):
    if preconditioner is not None:
        assert inv_preconditioner is not None
        vol_recon_HU_precond = inv_preconditioner(vol_init_HU)
        vol_recon_HU_precond = vol_recon_HU_precond.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([vol_recon_HU_precond], lr=lr)
    else:
        vol_recon_HU = vol_init_HU.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([vol_recon_HU], lr=lr)


    if warmup_iters is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda it: min(1.0, float(it) / warmup_iters)
        )


    t0 = time.time()
    print(f"Starting MAP reconstruction  (prior λ = {prior_weight})")

    for it in range(1, num_iters + 1):
        optimizer.zero_grad()
        if preconditioner is not None:
            vol_recon_HU = preconditioner(vol_recon_HU_precond)
        ll = log_likelihood_fn(vol_recon_HU)
        lp = log_prior_fn(vol_recon_HU)
        loss = ll + prior_weight * lp
        loss.backward()
        optimizer.step()

        if warmup_iters is not None:
            scheduler.step()
            

        if verbose:
            if it % print_every == 0 or it == 1:
                dt = time.time() - t0
                print(f"Iter {it:04d}  data={ll.item():.4e}  prior={prior_weight * lp.item():.4e}  "
                    f"loss={loss.item():.4e}   ({dt/60:.1f} min)")
                t0 = time.time()

    if preconditioner is not None:
        vol_recon_HU = preconditioner(vol_recon_HU_precond)

    return vol_recon_HU