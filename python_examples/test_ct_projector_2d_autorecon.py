#!/usr/bin/env python
import math
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from ct_laboratory.ct_projector_2d_module import CTProjector2DModule

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


def build_circular_phantom(n_row, n_col, center_offset=(-20, 10), radius=50.0):
    """Creates a 2D circular phantom in a [n_row, n_col] image."""
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    row_center = (n_row - 1) / 2.0 + center_offset[0]
    col_center = (n_col - 1) / 2.0 + center_offset[1]
    for row in range(n_row):
        for col in range(n_col):
            dist2 = (row - row_center)**2 + (col - col_center)**2
            if dist2 < radius**2:
                phantom[row, col] = 1.0
    return phantom

def build_fanbeam_rays(center_xy, n_view, n_det, source_distance, detector_distance, det_spacing):
    """Generates source & detector positions for a 2D fan-beam CT system."""
    cx, cy = center_xy
    ds, dd = source_distance, detector_distance
    angles = torch.arange(0, n_view) * (2 * math.pi / n_view)
    all_src, all_dst = [], []
    for theta in angles:
        sx, sy = cx + ds * math.cos(theta), cy + ds * math.sin(theta)
        dx_center, dy_center = cx - dd * math.cos(theta), cy - dd * math.sin(theta)
        # Perpendicular vector to the ray
        perp_x, perp_y = -(dy_center - sy), (dx_center - sx)
        norm_len = math.sqrt(perp_x**2 + perp_y**2)
        if norm_len < 1e-12:
            continue
        perp_x /= norm_len
        perp_y /= norm_len
        mid_i = (n_det - 1) / 2.0
        for i in range(n_det):
            offset = (i - mid_i) * det_spacing
            cell_x = dx_center + offset * perp_x
            cell_y = dy_center + offset * perp_y
            all_src.append([sx, sy])
            all_dst.append([cell_x, cell_y])
    return torch.tensor(all_src, dtype=torch.float32), torch.tensor(all_dst, dtype=torch.float32)

def main():
    # Setup phantom and geometry parameters
    n_row, n_col = 256, 256
    n_view = 360
    n_det = 400
    ds, dd = 200.0, 200.0
    det_spacing = 1.0

    # Build ground truth phantom (no grad tracking)
    phantom_gt = build_circular_phantom(n_row, n_col, center_offset=(-20, 10), radius=50.0)
    # Coordinate transform: (row,col) -> (x,y)
    A = torch.eye(2, dtype=torch.float32)
    row_mid, col_mid = (n_row - 1) / 2.0, (n_col - 1) / 2.0
    b = torch.tensor([-row_mid, -col_mid], dtype=torch.float32)
    src, dst = build_fanbeam_rays(center_xy=(0, 0), n_view=n_view, n_det=n_det,
                                  source_distance=ds, detector_distance=dd, det_spacing=det_spacing)

    # Set device (the module supports both CPU and GPU regardless of backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom_gt = phantom_gt.to(device)
    A = A.to(device)
    b = b.to(device)
    src = src.to(device)
    dst = dst.to(device)

    # Use the "torch" backend for this example (custom "cuda" extension could be used by setting backend='cuda')
    backend = 'torch'
    backend = 'cuda'

    # Build the projector module (which precomputes intersections)
    projector = CTProjector2DModule(n_row=n_row, n_col=n_col, M=A, b=b, src=src, dst=dst, backend=backend)
    projector.to(device)

    # Compute ground truth sinogram without grad tracking and add noise
    with torch.no_grad():
        sinogram_gt = projector(phantom_gt)
        noise_std = 10.0
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)

    # Set up the reconstruction: initialize the image to be optimized
    # Start with zeros then enable grad
    phantom_recon = torch.zeros(n_row, n_col, dtype=torch.float32, device=device)
    phantom_recon.requires_grad_()

    optimizer = torch.optim.Adam([phantom_recon], lr=0.1)
    # optimizer = torch.optim.SGD([phantom_recon], lr=0.1)
    mse_loss = torch.nn.MSELoss()

    num_iters = 200
    recon_history = []
    sino_pred_history = []
    loss_history = []

    # Precompute color limits based on ground truth phantom and noisy sinogram
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
        # Save current reconstruction and predicted sinogram for animation (detach and clone)
        recon_history.append(phantom_recon.detach().cpu().clone())
        sino_pred_history.append(sinogram_pred.detach().cpu().clone())

        if (it+1) % 1 == 0:
            print(f"Iteration {it+1}/{num_iters}: Loss = {loss.item():.6f}")
    end_time = time.time()
    print(f"Reconstruction completed in {end_time - start_time:.2f} seconds.")

    # Create a matplotlib animation with 4 subplots:
    # 1: Ground truth phantom (static)
    # 2: Noisy sinogram (static)
    # 3: Reconstructed phantom (updates)
    # 4: Predicted sinogram (updates)
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax_phantom_gt = axs[0, 0]
    ax_sinogram_gt = axs[0, 1]
    ax_recon = axs[1, 0]
    ax_sino_pred = axs[1, 1]

    im_phantom_gt = ax_phantom_gt.imshow(phantom_gt.cpu(), cmap='gray', origin='lower',
                                          vmin=phantom_vmin, vmax=phantom_vmax)
    ax_phantom_gt.set_title("Ground Truth Phantom")
    im_sinogram_gt = ax_sinogram_gt.imshow(sinogram_noisy.view(n_view, n_det).cpu(), cmap='gray', origin='lower',
                                            vmin=sino_vmin, vmax=sino_vmax, aspect='auto')
    ax_sinogram_gt.set_title("Noisy Sinogram")
    im_recon = ax_recon.imshow(recon_history[0], cmap='gray', origin='lower',
                               vmin=phantom_vmin, vmax=phantom_vmax)
    ax_recon.set_title("Reconstructed Phantom")
    im_sino_pred = ax_sino_pred.imshow(sino_pred_history[0].view(n_view, n_det), cmap='gray', origin='lower',
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
    # Save animation as mp4 (requires ffmpeg installed)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='ct_lab'), bitrate=1800)
    ani.save(f"{output_dir}/auto_recon_fanbeam_2d.mp4", writer=writer)
    print("Animation saved as auto_recon_animation.mp4")
    
    plt.show()

if __name__ == "__main__":
    main()
