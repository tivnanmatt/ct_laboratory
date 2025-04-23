#!/usr/bin/env python
"""
Demonstration of a 3D static gantry reconstruction.
We create a 3D spherical phantom [128,128,16], build a UniformStaticCTProjector3D 
with (n_source=200, n_module=48, each module=48x48), forward-project to get a sinogram,
add noise, then iteratively reconstruct using PyTorch autograd + Adam.
We display an animation of 4 subplots:
   (1) Ground truth volume (mid slice),
   (2) Noisy sinogram (for frame=0, module=0),
   (3) Reconstructed volume (mid slice),
   (4) Predicted sinogram (same frame=0, module=0).
"""

import math, time, os
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ct_laboratory.staticct_projector_3d import UniformStaticCTProjector3D

def build_3d_sphere(n_x, n_y, n_z, radius=20.0, center_offset=(0,0,0)):
    vol = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    cx = (n_x - 1)*0.5 + center_offset[0]
    cy = (n_y - 1)*0.5 + center_offset[1]
    cz = (n_z - 1)*0.5 + center_offset[2]
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                dx = i - cx
                dy = j - cy
                dz = k - cz
                if dx*dx + dy*dy + dz*dz < radius*radius:
                    vol[i,j,k] = 1.0
    return vol

def main():
    # ---------------------------------------------------------
    # 1) Setup volume + geometry
    # ---------------------------------------------------------
    n_x, n_y, n_z = 128, 128, 16
    phantom_gt = build_3d_sphere(n_x, n_y, n_z, radius=30.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom_gt = phantom_gt.to(device)

    # We'll define a uniform static geometry:
    n_source = 200
    n_frame = n_source
    n_module = 48
    det_nx_per_module = 48
    det_ny_per_module = 16
    det_spacing_x = 1.0
    det_spacing_y = 2.0
    source_radius = 380.00
    module_radius = 366.70
    source_z_offset = 0.0
    module_z_offset = 0.0

    # We assume n_frame = n_source => 1 source active per frame
    M_list = []
    b_list = []
    active_src = torch.zeros(n_source, n_source, dtype=torch.bool)
    for i in range(n_source):
        M_ = torch.eye(3, dtype=torch.float32)
        b_ = torch.zeros(3, dtype=torch.float32)
        M_list.append(M_)
        b_list.append(b_)
        active_src[i, i] = True
    M_gantry = torch.stack(M_list, dim=0).to(device)  # [n_frame,3,3]
    b_gantry = torch.stack(b_list, dim=0).to(device)

    M_3d = torch.eye(3, dtype=torch.float32)
    b_3d = torch.zeros(3, dtype=torch.float32)
    b_3d[0] = -n_x*0.5
    b_3d[1] = -n_y*0.5
    b_3d[2] = -n_z*0.5

    projector = UniformStaticCTProjector3D(
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        n_source=n_source,
        source_radius=source_radius,
        source_z_offset=source_z_offset,
        n_module=n_module,
        module_radius=module_radius,
        det_nx_per_module=det_nx_per_module,
        det_ny_per_module=det_ny_per_module,
        det_spacing_x=det_spacing_x,
        det_spacing_y=det_spacing_y,
        module_z_offset=module_z_offset,
        M_gantry=M_gantry,
        b_gantry=b_gantry,
        active_sources=active_src,
        M=M_3d,
        b=b_3d,
        backend="cuda",
        device=device
    ).to(device)

    # -----------------------------------------
    # 2) Forward project ground-truth phantom + add noise
    # ---------------------------------------------------------
    with torch.no_grad():
        sinogram_gt = projector(phantom_gt)
        noise_std = 0.0
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)

    # We'll reshape for the "display" frames in the animation:
    # shape => [n_frame, n_module, det_nx_per_module, det_ny_per_module]
    sinogram_4d = sinogram_noisy.view(
        n_source, n_module, det_nx_per_module, det_ny_per_module
    )

    # ---------------------------------------------------------
    # 3) Set up reconstruction problem
    # ---------------------------------------------------------
    volume_recon = torch.zeros_like(phantom_gt, requires_grad=True)
    optimizer = torch.optim.Adam([volume_recon], lr=0.1)
    mse_loss = torch.nn.MSELoss()

    num_iters = 120
    recon_history = []
    sino_pred_history = []
    loss_history = []

    # We'll track a single "view" of the sinogram for display:
    # We'll pick frame=0, module=0 => shape [48,48].
    sino_gt_view = sinogram_4d[0,0].clone().cpu()
    sino_vmin, sino_vmax = sino_gt_view.min().item(), sino_gt_view.max().item()

    # Also track the mid Z-slice of the phantom for display
    mid_z = n_z//2 + 7
    phantom_vmin = phantom_gt.min().item()
    phantom_vmax = phantom_gt.max().item()

    # ---------------------------------------------------------
    # 4) Iterative reconstruction
    # ---------------------------------------------------------
    start_time = time.time()
    for it in range(num_iters):
        optimizer.zero_grad()
        sino_pred = projector(volume_recon)
        loss = mse_loss(sino_pred, sinogram_noisy)
        loss.backward()
        optimizer.step()

        recon_history.append(volume_recon.detach().cpu().clone())
        # Reshape the predicted sinogram to 4D => we can extract the same view
        sino_pred_4d = sino_pred.view(n_source, n_module, det_nx_per_module, det_ny_per_module)
        # store just that single frame=0, module=0
        sino_pred_view = sino_pred_4d[0,0].detach().cpu().clone()

        sino_pred_history.append(sino_pred_view)
        loss_history.append(loss.item())

        if (it+1) % 10 == 0:
            print(f"Iter {it+1}/{num_iters}, loss={loss.item():.6f}")
    end_time = time.time()
    print(f"Reconstruction completed in {end_time - start_time:.1f} s.")

    # ---------------------------------------------------------
    # 5) Build Matplotlib animation of the reconstruction
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    ax_vol_gt     = axs[0,0]
    ax_sino_gt    = axs[0,1]
    ax_vol_recon  = axs[1,0]
    ax_sino_pred  = axs[1,1]

    im_vol_gt = ax_vol_gt.imshow(phantom_gt[:,:,mid_z].cpu(), cmap='gray', origin='lower',
                                 vmin=phantom_vmin, vmax=phantom_vmax)
    ax_vol_gt.set_title("Ground Truth Volume (mid slice)")

    im_sino_gt = ax_sino_gt.imshow(sino_gt_view, cmap='gray', origin='lower',
                                   vmin=sino_vmin, vmax=sino_vmax)
    ax_sino_gt.set_title("Noisy Sinogram (frame=0, module=0)")

    im_vol_recon = ax_vol_recon.imshow(recon_history[0][:,:,mid_z], cmap='gray', origin='lower',
                                       vmin=phantom_vmin, vmax=phantom_vmax)
    ax_vol_recon.set_title("Reconstructed Volume (mid slice)")

    im_sino_pred = ax_sino_pred.imshow(sino_pred_history[0], cmap='gray', origin='lower',
                                       vmin=sino_vmin, vmax=sino_vmax)
    ax_sino_pred.set_title("Predicted Sinogram (view=0,mod=0)")

    plt.tight_layout()

    def update(frame_idx):
        print(f"Frame {frame_idx+1}/{num_iters}")
        im_vol_recon.set_data(recon_history[frame_idx][:,:,mid_z])
        im_sino_pred.set_data(sino_pred_history[frame_idx])
        fig.suptitle(f"Iter {frame_idx+1}/{num_iters}, Loss={loss_history[frame_idx]:.6f}", fontsize=16)
        return [im_vol_recon, im_sino_pred]

    ani = animation.FuncAnimation(fig, update, frames=num_iters, interval=200, blit=False)

    # Save animation as mp4
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='ct_lab'), bitrate=1800)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "staticct_3d_autorecon.mp4")
    ani.save(out_path, writer=writer)
    print(f"Animation saved => {out_path}")

    plt.show()

if __name__=="__main__":
    main()
