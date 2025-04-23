#!/usr/bin/env python
import math
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ct_laboratory.ct_projector_3d_module import CTProjector3DModule

# get current file name and parse directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")


def build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=30.0):
    """Creates a 3D volume with a sphere inside."""
    vol = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    cx, cy, cz = (n_x - 1) / 2 + center[0], (n_y - 1) / 2 + center[1], (n_z - 1) / 2 + center[2]
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                dist2 = (i - cx)**2 + (j - cy)**2 + (k - cz)**2
                if dist2 <= radius**2:
                    vol[i, j, k] = 1.0
    return vol

def build_conebeam_geometry_3d(n_x, n_y, n_z, n_view=90, det_nx=64, det_ny=64,
                               ds=200.0, dd=200.0, det_spacing=1.0):
    """
    Generates a 3D cone-beam projection geometry rotating around the center of the volume.
    Assumes voxel indices are directly in world coordinates (A=I, b=0).
    """
    all_src, all_dst = [], []
    cx, cy, cz = (n_x - 1) / 2, (n_y - 1) / 2, (n_z - 1) / 2
    angles = torch.linspace(0, 2 * math.pi - 2 * math.pi / n_view, n_view)
    for theta in angles:
        # Source position (rotating around volume center)
        sx = cx + ds * math.cos(theta)
        sy = cy + ds * math.sin(theta)
        sz = cz  # keep z centered
        # Detector center position
        dx_c = cx - dd * math.cos(theta)
        dy_c = cy - dd * math.sin(theta)
        dz_c = cz  # keep detector centered in z
        # Detector plane axes
        perp_x, perp_y = -(dy_c - sy), (dx_c - sx)
        norm_len = math.sqrt(perp_x**2 + perp_y**2)
        if norm_len < 1e-9:
            continue
        perp_x /= norm_len
        perp_y /= norm_len
        # Use vertical axis in z
        v_x, v_y, v_z = 0.0, 0.0, 1.0
        mid_u, mid_v = (det_nx - 1) / 2, (det_ny - 1) / 2
        for iu in range(det_nx):
            for iv in range(det_ny):
                offset_u = (iu - mid_u) * det_spacing
                offset_v = (iv - mid_v) * det_spacing
                cell_x = dx_c + offset_u * perp_x + offset_v * v_x
                cell_y = dy_c + offset_u * perp_y + offset_v * v_y
                cell_z = dz_c + offset_u * 0.0    + offset_v * v_z
                all_src.append([sx, sy, sz])
                all_dst.append([cell_x, cell_y, cell_z])
    return torch.tensor(all_src, dtype=torch.float32), torch.tensor(all_dst, dtype=torch.float32)

def main():
    # Geometry parameters
    n_x, n_y, n_z = 64, 64, 64
    n_view = 120
    det_nx, det_ny = 100, 100
    ds, dd = 200.0, 200.0
    det_spacing = 1.0

    # Build ground truth volume (spherical phantom)
    volume_gt = build_3d_sphere(n_x, n_y, n_z, center=(0,0,0), radius=20.0)
    
    # Transform: (i,j,k) -> (x,y,z) : use identity, b=0
    A = torch.eye(3, dtype=torch.float32)
    b = torch.zeros(3, dtype=torch.float32)
    
    # Build cone-beam geometry
    src, dst = build_conebeam_geometry_3d(n_x, n_y, n_z, n_view, det_nx, det_ny, ds, dd, det_spacing)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume_gt = volume_gt.to(device)
    A = A.to(device)
    b = b.to(device)
    src = src.to(device)
    dst = dst.to(device)
    
    backend = 'cuda'  # use torch backend for the example
    
    # Build the 3D projector module (precomputes intersections)
    projector = CTProjector3DModule(n_x=n_x, n_y=n_y, n_z=n_z, M=A, b=b, src=src, dst=dst, backend=backend)
    projector.to(device)
    
    # Compute ground truth sinogram (without grad) and add noise.
    with torch.no_grad():
        sinogram_gt = projector(volume_gt)
        noise_std = 10.0
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)
    
    # Set up the reconstruction problem: the volume is the parameter.
    # Initialize with zeros.
    volume_recon = torch.zeros(n_x, n_y, n_z, dtype=torch.float32, device=device)
    volume_recon.requires_grad_()
    
    optimizer = torch.optim.Adam([volume_recon], lr=0.1)
    mse_loss = torch.nn.MSELoss()
    
    num_iters = 200
    recon_history = []
    sino_pred_history = []
    loss_history = []
    
    # For display: we use the mid-slice along z (i.e. index n_z//2)
    mid_z = n_z // 2
    # Also extract one view (e.g., view index 0) from the sinogram.
    # sinogram shape: [n_ray] if volume is 3D. Reshape to [n_view, det_nx, det_ny]
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
        if (it+1) % 1 == 0:
            print(f"Iteration {it+1}/{num_iters}: Loss = {loss.item():.6f}")
    end_time = time.time()
    print(f"3D Reconstruction completed in {end_time - start_time:.2f} seconds.")
    
    # Prepare animation with 4 subplots:
    # Top left: Ground truth volume (mid slice)
    # Top right: Noisy sinogram (view 0)
    # Bottom left: Reconstructed volume (mid slice)
    # Bottom right: Predicted sinogram (view 0)
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
    ani.save(f"{output_dir}/auto_recon_3d_animation.mp4", writer=writer)
    print("Animation saved as auto_recon_3d_animation.mp4")
    plt.show()

if __name__ == "__main__":
    main()
