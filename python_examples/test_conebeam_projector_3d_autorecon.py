#!/usr/bin/env python
import math
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Use the new high-level module instead of ct_projector_3d_module.
from ct_laboratory.conebeam_projector_3d import ConeBeam3DProjector

# Get current file directory and define an output directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

def build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=30.0):
    """Creates a 3D volume with a sphere inside."""
    vol = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    cx = (n_x - 1) / 2.0 + center[0]
    cy = (n_y - 1) / 2.0 + center[1]
    cz = (n_z - 1) / 2.0 + center[2]
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                if (i - cx)**2 + (j - cy)**2 + (k - cz)**2 < radius**2:
                    vol[i, j, k] = 1.0
    return vol

def main():
    # --- Geometry parameters ---
    n_x, n_y, n_z = 64, 64, 64
    n_view = 120
    det_nx, det_ny = 100, 100
    sid = 200.0   # Source-to-Isocenter distance
    sdd = 200.0 + sid  # Source-to-Detector distance
    det_spacing = 1.0
    voxel_size = 1.0

    # --- Build phantom volume ---
    volume_gt = build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=20.0)

    # --- Use the new ConeBeam3DProjector ---
    backend = 'cuda' if torch.cuda.is_available() else 'torch'
    projector = ConeBeam3DProjector(
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        n_view=n_view,
        det_nx=det_nx,
        det_ny=det_ny,
        sid=sid,
        sdd=sdd,
        det_spacing=det_spacing,
        voxel_size=voxel_size,
        backend=backend
    )

    # --- Set device and move data ---
    device = torch.device("cuda" if backend == "cuda" else "cpu")
    volume_gt = volume_gt.to(device)
    projector.to(device)

    # --- Compute ground truth sinogram and add noise ---
    with torch.no_grad():
        sinogram_gt = projector(volume_gt)
        noise_std = 10.0
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)

    # --- Set up reconstruction: initialize volume to zeros ---
    volume_recon = torch.zeros(n_x, n_y, n_z, dtype=torch.float32, device=device)
    volume_recon.requires_grad_()

    optimizer = torch.optim.Adam([volume_recon], lr=0.1)
    mse_loss = torch.nn.MSELoss()

    num_iters = 200
    recon_history = []
    sino_pred_history = []
    loss_history = []

    # For display, use mid-slice along z and view 0 from the sinogram.
    mid_z = n_z // 2
    sino_gt_reshaped = sinogram_noisy.view(n_view, det_nx, det_ny)
    sino_vmin, sino_vmax = sino_gt_reshaped.min().item(), sino_gt_reshaped.max().item()
    volume_vmin, volume_vmax = volume_gt.min().item(), volume_gt.max().item()

    start_time = time.time()
    for it in range(num_iters):
        optimizer.zero_grad()
        sino_pred = projector(volume_recon)
        loss = mse_loss(sino_pred, sinogram_noisy)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        recon_history.append(volume_recon.detach().cpu().clone())
        sino_pred_history.append(sino_pred.detach().cpu().clone())
        print(f"Iteration {it+1}/{num_iters}: Loss = {loss.item():.6f}")
    end_time = time.time()
    print(f"3D Reconstruction completed in {end_time - start_time:.2f} seconds.")

    # --- Animation: 4 subplots ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_vol_gt = axs[0, 0]
    ax_sino_gt = axs[0, 1]
    ax_vol_recon = axs[1, 0]
    ax_sino_pred = axs[1, 1]

    im_vol_gt = ax_vol_gt.imshow(volume_gt[:, :, mid_z].cpu(), cmap='gray', origin='lower',
                                 vmin=volume_vmin, vmax=volume_vmax)
    ax_vol_gt.set_title("Ground Truth Volume (mid slice)")
    im_sino_gt = ax_sino_gt.imshow(sino_gt_reshaped[0].cpu(), cmap='gray', origin='lower',
                                   vmin=sino_vmin, vmax=sino_vmax)
    ax_sino_gt.set_title("Noisy Sinogram (view 0)")
    im_vol_recon = ax_vol_recon.imshow(recon_history[0][:, :, mid_z], cmap='gray', origin='lower',
                                       vmin=volume_vmin, vmax=volume_vmax)
    ax_vol_recon.set_title("Reconstructed Volume (mid slice)")
    sino_pred0 = sino_pred_history[0].view(n_view, det_nx, det_ny)
    im_sino_pred = ax_sino_pred.imshow(sino_pred0[0].cpu(), cmap='gray', origin='lower',
                                        vmin=sino_vmin, vmax=sino_vmax)
    ax_sino_pred.set_title("Predicted Sinogram (view 0)")

    for ax in axs.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.tight_layout()

    def update(frame):
        print(f"Frame {frame+1}/{num_iters}")
        im_vol_recon.set_data(recon_history[frame][:, :, mid_z])
        sino_pred_frame = sino_pred_history[frame].view(n_view, det_nx, det_ny)
        im_sino_pred.set_data(sino_pred_frame[0])
        fig.suptitle(f"Iteration {frame+1}/{num_iters} Loss: {loss_history[frame]:.6f}", fontsize=16)
        return im_vol_recon, im_sino_pred

    ani = animation.FuncAnimation(fig, update, frames=num_iters, interval=100, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='ct_lab'), bitrate=1800)
    ani.save(os.path.join(output_dir, "auto_recon_conebeam_3d_module.mp4"), writer=writer)
    print("Animation saved as auto_recon_conebeam_3d.mp4")
    plt.show()

if __name__ == "__main__":
    main()
