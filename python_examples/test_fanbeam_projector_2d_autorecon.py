#!/usr/bin/env python
import math
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

# Use the new high‐level module instead of ct_projector_2d_module.
from ct_laboratory.fanbeam_projector_2d import FanBeam2DProjector

# Create an output directory relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

def build_circular_phantom(n_row, n_col, center_offset=(-20, 10), radius=50.0):
    """Creates a 2D circular phantom in a [n_row, n_col] image."""
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    row_center = (n_row - 1) / 2.0 + center_offset[0]
    col_center = (n_col - 1) / 2.0 + center_offset[1]
    for row in range(n_row):
        for col in range(n_col):
            if (row - row_center)**2 + (col - col_center)**2 < radius**2:
                phantom[row, col] = 1.0
    return phantom

def main():
    # --- Parameters ---
    n_row, n_col = 256, 256
    n_view = 360
    n_det = 400
    sid = 200.0    # Source-to-Isocenter distance
    sdd = 200.0 + sid  # Source-to-Detector distance (sdd = sid + detector distance)
    det_spacing = 1.0
    pixel_size = 1.0

    # --- Build phantom ---
    phantom_gt = build_circular_phantom(n_row, n_col, center_offset=(-20, 10), radius=50.0)

    # --- Use the new FanBeam2DProjector ---
    # The new module internally computes src and dst given standard fan-beam parameters.
    backend = 'cuda' if torch.cuda.is_available() else 'torch'
    projector = FanBeam2DProjector(
        n_row=n_row,
        n_col=n_col,
        n_view=n_view,
        n_det=n_det,
        sid=sid,
        sdd=sdd,
        det_spacing=det_spacing,
        pixel_size=pixel_size,
        backend=backend
    )

    # --- Set device and move data ---
    device = torch.device("cuda" if backend == "cuda" else "cpu")
    phantom_gt = phantom_gt.to(device)
    projector.to(device)

    # --- Compute sinogram with ground truth phantom ---
    with torch.no_grad():
        sinogram_gt = projector(phantom_gt)
        noise_std = 10.0
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)

    # --- Reconstruction: optimize an image initialized to zeros ---
    phantom_recon = torch.zeros(n_row, n_col, dtype=torch.float32, device=device)
    phantom_recon.requires_grad_()

    optimizer = torch.optim.Adam([phantom_recon], lr=0.1)
    mse_loss = torch.nn.MSELoss()

    num_iters = 200
    recon_history = []
    sino_pred_history = []
    loss_history = []

    # For display, set color limits
    phantom_vmin, phantom_vmax = phantom_gt.min().item(), phantom_gt.max().item()
    sino_vmin, sino_vmax = sinogram_noisy.min().item(), sinogram_noisy.max().item()

    start_time = time.time()
    for it in range(num_iters):
        optimizer.zero_grad()
        sinogram_pred = projector(phantom_recon)
        loss = mse_loss(sinogram_pred, sinogram_noisy)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        recon_history.append(phantom_recon.detach().cpu().clone())
        sino_pred_history.append(sinogram_pred.detach().cpu().clone())

        if (it+1) % 20 == 0:
            print(f"Iteration {it+1}/{num_iters}: Loss = {loss.item():.6f}")
    end_time = time.time()
    print(f"Reconstruction completed in {end_time - start_time:.2f} seconds.")

    # --- Create animation ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_phantom_gt = axs[0, 0]
    ax_sinogram_gt = axs[0, 1]
    ax_recon = axs[1, 0]
    ax_sino_pred = axs[1, 1]

    im_phantom_gt = ax_phantom_gt.imshow(phantom_gt.cpu(), cmap='gray', origin='lower',
                                          vmin=phantom_vmin, vmax=phantom_vmax)
    ax_phantom_gt.set_title("Ground Truth Phantom")
    im_sinogram_gt = ax_sinogram_gt.imshow(sinogram_noisy.view(n_view, n_det).cpu(),
                                            cmap='gray', origin='lower',
                                            vmin=sino_vmin, vmax=sino_vmax, aspect='auto')
    ax_sinogram_gt.set_title("Noisy Sinogram")
    im_recon = ax_recon.imshow(recon_history[0], cmap='gray', origin='lower',
                               vmin=phantom_vmin, vmax=phantom_vmax)
    ax_recon.set_title("Reconstructed Phantom")
    im_sino_pred = ax_sino_pred.imshow(sino_pred_history[0].view(n_view, n_det).cpu(),
                                        cmap='gray', origin='lower',
                                        vmin=sino_vmin, vmax=sino_vmax, aspect='auto')
    ax_sino_pred.set_title("Predicted Sinogram")

    for ax in axs.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.tight_layout()

    def update(frame):
        print(f"Frame {frame+1}/{num_iters}")
        im_recon.set_data(recon_history[frame])
        im_sino_pred.set_data(sino_pred_history[frame].view(n_view, n_det))
        fig.suptitle(f"Iteration {frame+1}/{num_iters} Loss: {loss_history[frame]:.6f}", fontsize=16)
        return im_recon, im_sino_pred

    ani = animation.FuncAnimation(fig, update, frames=num_iters, interval=100, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='ct_lab'), bitrate=1800)
    ani.save(os.path.join(output_dir, "auto_recon_fanbeam_2d_module.mp4"), writer=writer)
    print("Animation saved as auto_recon_fanbeam_2d_module.mp4")
    
    plt.show()

if __name__ == "__main__":
    main()
