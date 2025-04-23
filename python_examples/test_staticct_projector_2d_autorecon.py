#!/usr/bin/env python
import math
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Import the 2D Uniform Static CT projector from your package.
from ct_laboratory.staticct_projector_2d import UniformStaticCTProjector2D

def build_circular_phantom(n_row, n_col, radius=40.0):
    """
    Create a 2D circular phantom in an [n_row, n_col] image.
    """
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    row_center = (n_row - 1) / 2.0
    col_center = (n_col - 1) / 2.0
    for r in range(n_row):
        for c in range(n_col):
            if (r - row_center)**2 + (c - col_center)**2 < radius**2:
                phantom[r, c] = 1.0
    return phantom

def main():
    # Set up an output directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Image dimensions
    n_row, n_col = 256, 256

    # Build the ground truth phantom.
    phantom_gt = build_circular_phantom(n_row, n_col, radius=50.0)

    import pydicom
    dicom_dir = '/mnt/AXIS02_share/rsna-intracranial-hemorrhage-detection/stage_2_test/'
    # get the file names there ending in dcm
    dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    # read the first file
    dcm_file = dcm_files[2]
    dcm_path = os.path.join(dicom_dir, dcm_file)
    dcm_data = pydicom.dcmread(dcm_path)
    # get the pixel array
    pixel_array = dcm_data.pixel_array
    pixel_array[pixel_array<0] = 0
    # down sample 2x
    pixel_array = pixel_array[::2, ::2]
    # flip the col
    pixel_array = pixel_array[::-1, :]
    # deal with non contiguous memory
    pixel_array = pixel_array.copy()
    # convert to torch tensor
    phantom_gt = torch.tensor(pixel_array, dtype=torch.float32)
    # phantom_gt = phantom_gt + 1000
    phantom_gt = phantom_gt * 0.17 / 1000

    # -------------------------------------------------------------------
    # Define 2D Static CT geometry parameters.
    # For a uniform static CT, we assume:
    #  - n_source sources arranged around a circle.
    #  - n_module detector modules arranged on another circle.
    #  - Each module has det_n_col detector columns.
    #  - Per-frame (gantry) transforms (M_gantry, b_gantry) are provided,
    #    with each frame activating one source.
    # -------------------------------------------------------------------
    n_source = 200       # number of sources around the circle
    n_frame = n_source   # one frame per source (i.e. only one source active per frame)
    n_module = 48       # number of detector modules
    det_n_col = 48      # detector columns per module
    det_spacing = 1.0   # detector spacing in mm
    source_radius = 380.00  # source-to-center distance (mm)
    module_radius = 366.70  # module-to-center distance (mm)

    # Build per-frame gantry transforms and active source mask.
    M_list = []
    b_list = []
    active_sources = torch.zeros(n_frame, n_source, dtype=torch.bool)
    for i in range(n_frame):
        # Here we use an identity transform (no rotation or translation).
        M_ = torch.eye(2, dtype=torch.float32)
        b_ = torch.zeros(2, dtype=torch.float32)
        b_[0] = n_col / 2.0
        b_[1] = n_row / 2.0
        M_list.append(M_)
        b_list.append(b_)
        # Activate only the i-th source in frame i.
        active_sources[i, i] = True

    M_gantry = torch.stack(M_list, dim=0)  # shape: [n_frame, 2, 2]
    b_gantry = torch.stack(b_list, dim=0)    # shape: [n_frame, 2]

    M_2d = torch.eye(2, dtype=torch.float32)
    b_2d = torch.zeros(2, dtype=torch.float32)

    projector = UniformStaticCTProjector2D(
        n_row=n_row,
        n_col=n_col,
        M=M_2d,
        b=b_2d,
        n_source=n_source,
        source_radius=source_radius,
        n_module=n_module,
        module_radius=module_radius,
        det_n_col=det_n_col,
        det_spacing=det_spacing,
        M_gantry=M_gantry,
        b_gantry=b_gantry,
        active_sources=active_sources,
        backend="cuda"
    )

    # Set device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom_gt = phantom_gt.to(device)
    projector.to(device)

    # Compute the sinogram from the ground truth phantom and add noise.
    with torch.no_grad():
        sinogram_gt = projector(phantom_gt)
        noise_std = 0.5
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)

    # Set up the reconstruction: initialize the image (phantom) to zeros.
    phantom_recon = torch.zeros(n_row, n_col, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([phantom_recon], lr=0.1)
    mse_loss = torch.nn.MSELoss()

    num_iters = 200
    recon_history = []
    sino_pred_history = []
    loss_history = []

    # For visualization, reshape the 1D sinogram into a 2D image.
    # Total number of rays = n_frame * (n_module * det_n_col)
    sinogram_noisy_2d = sinogram_noisy.view(n_frame, n_module * det_n_col).cpu()
    sino_vmin, sino_vmax = sinogram_noisy_2d.min().item(), sinogram_noisy_2d.max().item()
    phantom_vmin, phantom_vmax = phantom_gt.min().item(), phantom_gt.max().item()

    # Iterative reconstruction loop.
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
        if (it+1) % 10 == 0:
            print(f"Iteration {it+1}/{num_iters}: Loss = {loss.item():.6f}")
    end_time = time.time()
    print(f"Reconstruction completed in {end_time - start_time:.2f} seconds.")

    # Create an animation with 4 subplots:
    # 1. Ground Truth Phantom (static)
    # 2. Noisy Sinogram (static)
    # 3. Reconstructed Phantom (updates)
    # 4. Predicted Sinogram (updates)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_phantom_gt = axs[0, 0]
    ax_sinogram_gt = axs[0, 1]
    ax_recon = axs[1, 0]
    ax_sino_pred = axs[1, 1]

    im_phantom_gt = ax_phantom_gt.imshow(phantom_gt.cpu(), cmap='gray', origin='lower',
                                          vmin=phantom_vmin, vmax=phantom_vmax)
    ax_phantom_gt.set_title("Ground Truth Phantom")
    # sinogram_noisy_2d = sinogram_noisy_2d.reshape(n_frame, n_module, det_n_col)
    # sinogram_noisy_2d = torch.flip(sinogram_noisy_2d, [2])
    # sinogram_noisy_2d = sinogram_noisy_2d.reshape(n_frame, n_module * det_n_col)
    im_sinogram_gt = ax_sinogram_gt.imshow(sinogram_noisy_2d, cmap='gray', origin='lower',
                                            vmin=sino_vmin, vmax=sino_vmax, aspect='auto')
    ax_sinogram_gt.set_title("Noisy Sinogram")
    im_recon = ax_recon.imshow(recon_history[0], cmap='gray', origin='lower',
                               vmin=phantom_vmin, vmax=phantom_vmax)
    ax_recon.set_title("Reconstructed Phantom")
    sino_pred_2d = sino_pred_history[0].view(n_frame, n_module * det_n_col)
    im_sino_pred = ax_sino_pred.imshow(sino_pred_2d, cmap='gray', origin='lower',
                                        vmin=sino_vmin, vmax=sino_vmax, aspect='auto')
    ax_sino_pred.set_title("Predicted Sinogram")

    for ax in axs.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    plt.tight_layout()

    def update(frame):
        print(f"Frame {frame+1}/{num_iters}")
        im_recon.set_data(recon_history[frame])
        sino_frame = sino_pred_history[frame].reshape(n_frame, n_module, det_n_col)
        # flip the columns
        # sino_frame = torch.flip(sino_frame, [2])
        sino_frame = sino_frame.reshape(n_frame, n_module * det_n_col)
        # sino_frame = sino_pred_history[frame].view(n_frame, n_module * det_n_col)
        im_sino_pred.set_data(sino_frame)
        fig.suptitle(f"Iteration {frame+1}/{num_iters} Loss: {loss_history[frame]:.6f}", fontsize=16)
        return im_recon, im_sino_pred

    ani = animation.FuncAnimation(fig, update, frames=num_iters, interval=100, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='ct_lab'), bitrate=1800)
    ani.save(os.path.join(output_dir, "autorecon_staticct_2d.mp4"), writer=writer)
    print("Animation saved as autorecon_staticct_2d.mp4")
    plt.show()

if __name__ == "__main__":
    main()
