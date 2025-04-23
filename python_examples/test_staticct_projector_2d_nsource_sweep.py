#!/usr/bin/env python
"""
Static CT 2D reconstruction sweep over number of sources.

This script loads a DICOM phantom (using the same procedure as in the test_staticct_projector_2d_autorecon script),
and then for each number of sources (from 1 to 360) it builds the uniform static CT geometry, runs 200 iterations
of reconstruction (using Adam), and records the final RMSE and total reconstruction time. At the end it displays three figures:
  1. An array of images: the ground truth phantom and the reconstructions for 36, 72, 120, 180, and 360 sources.
  2. A plot of RMSE vs. number of sources.
  3. A plot of Time (s) vs. number of sources.
"""

import os, time, math
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydicom

# Import the Uniform Static 2D projector.
from ct_laboratory.staticct_projector_2d import UniformStaticCTProjector2D

# -------------------------------
# 1. Load the DICOM phantom
# -------------------------------
dicom_dir = '/mnt/AXIS02_share/rsna-intracranial-hemorrhage-detection/stage_2_test/'
dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
if len(dcm_files) == 0:
    raise RuntimeError("No DICOM files found in " + dicom_dir)
# Using the third file (index 2) as in the original test script.
dcm_file = dcm_files[2]
dcm_path = os.path.join(dicom_dir, dcm_file)
dcm_data = pydicom.dcmread(dcm_path)
pixel_array = dcm_data.pixel_array
# Remove negative values, downsample 2x, flip vertically, and copy to ensure contiguity.
pixel_array[pixel_array < 0] = 0
pixel_array = pixel_array[::2, ::2]
pixel_array = pixel_array[::-1, :]
pixel_array = pixel_array.copy()
# Convert to torch tensor (float32) and scale as in the original test.
phantom_gt = torch.tensor(pixel_array, dtype=torch.float32)
phantom_gt = phantom_gt * 0.17 / 1000.0

# Get image dimensions.
n_row, n_col = phantom_gt.shape

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
phantom_gt = phantom_gt.to(device)

# -------------------------------
# 2. Define reconstruction parameters
# -------------------------------
# Parameters for the static CT geometry.
n_module    = 48         # number of detector modules
det_n_col   = 48         # number of detector columns per module
det_spacing = 1.0        # detector pixel spacing (mm)
source_radius   = 380.0  # distance from center to source (mm)
module_radius   = 366.70 # distance from center to module (mm)
pixel_size      = 1.0     # scaling from (row,col) to (x,y)
num_iters       = 200     # iterations of reconstruction for each geometry

# For later plotting we want to save reconstructions for these source counts.
save_views   = [36, 72, 120, 180, 360]
saved_recons = {}      # store reconstructions for selected source counts
rmse_values  = []      # record RMSE for each n_source
source_counts = []     # list of n_source values
time_values  = []      # record total reconstruction time for each n_source

# -------------------------------
# 3. Set up output directory
# -------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 4. Loop over number of sources from 1 to 360
# -------------------------------
print("Starting sweep over number of sources...")
for n_source in range(1, 361):
    # For the static CT geometry, assume one frame per source.
    n_frame = n_source

    # Build per-frame gantry transforms (identity) and active source mask.
    M_list = [torch.eye(2, dtype=torch.float32) for _ in range(n_frame)]
    b_list = [torch.zeros(2, dtype=torch.float32) for _ in range(n_frame)]
    M_gantry = torch.stack(M_list, dim=0).to(device)  # [n_frame, 2, 2]
    b_gantry = torch.stack(b_list, dim=0).to(device)    # [n_frame, 2]
    active_sources = torch.eye(n_source, dtype=torch.bool).to(device)

    # Instantiate the UniformStaticCTProjector2D.
    projector = UniformStaticCTProjector2D(
        n_row=n_row,
        n_col=n_col,
        n_source=n_source,
        source_radius=source_radius,
        n_module=n_module,
        module_radius=module_radius,
        det_n_col=det_n_col,
        det_spacing=det_spacing,
        M_gantry=M_gantry,
        b_gantry=b_gantry,
        active_sources=active_sources,
        pixel_size=pixel_size,
        backend="cuda"  # Change to "torch" if preferred.
    ).to(device)

    # Compute the sinogram from phantom_gt and add noise.
    with torch.no_grad():
        sinogram_gt = projector(phantom_gt)
        noise_std = 0.5
        exposure_relative_to_360 = 360.0 / n_source
        noise_std /= exposure_relative_to_360 ** 0.5
        noise_std = 0.0
        sinogram_noisy = sinogram_gt + noise_std * torch.randn_like(sinogram_gt)

    # Initialize reconstruction image (phantom) as zeros.
    recon = torch.zeros(n_row, n_col, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([recon], lr=0.1)
    loss_fn = torch.nn.MSELoss()

    # Run reconstruction optimization for num_iters iterations while timing.
    t0 = time.perf_counter()
    for it in range(num_iters):
        optimizer.zero_grad()
        sinogram_pred = projector(recon)
        loss = loss_fn(sinogram_pred, sinogram_noisy)
        loss.backward()
        optimizer.step()
    t1 = time.perf_counter()
    time_values.append(t1 - t0)

    # Compute RMSE for this reconstruction.
    rmse = torch.sqrt(torch.mean((recon - phantom_gt) ** 2)).item()
    rmse_values.append(rmse)
    source_counts.append(n_source)

    # clean up the projector
    del projector

    # Save the reconstruction for selected source counts.
    if n_source in save_views:
        saved_recons[n_source] = recon.detach().cpu().numpy()

    if n_source %  1 == 0 or n_source == 1:
        print(f"n_source = {n_source:3d} | RMSE = {rmse:.4f}")

# -------------------------------
# 5. Plot final results
# -------------------------------

# Figure 1: Array of images (ground truth + reconstructions for selected source counts).
fig1, axs1 = plt.subplots(1, len(save_views) + 1, figsize=(15, 3))
axs1[0].imshow(phantom_gt.cpu().numpy(), cmap='gray', origin='lower')
axs1[0].set_title("Ground Truth")
axs1[0].axis('off')
for idx, ns in enumerate(save_views):
    axs1[idx+1].imshow(saved_recons[ns], cmap='gray', origin='lower')
    axs1[idx+1].set_title(f"{ns} sources")
    axs1[idx+1].axis('off')
fig1.suptitle("Reconstructions (200 iterations) for Selected Number of Sources")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "reconstructions_vs_views.png"), dpi=150)

# Figure 2: RMSE vs. number of sources.
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(source_counts, rmse_values, marker='o', linestyle='-', markersize=4)
ax2.set_xlabel("Number of Sources (and frames)")
ax2.set_ylabel("RMSE")
ax2.set_title("RMSE vs. Number of Sources")
# make it log y scale
ax2.set_yscale('log')
ax2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rmse_vs_sources.png"), dpi=150)

# Figure 3: Time vs. number of sources.
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(source_counts, time_values, marker='o', linestyle='-', markersize=4)
ax3.set_xlabel("Number of Sources (and frames)")
ax3.set_ylabel("Time (s)")
ax3.set_title("Time vs. Number of Sources")
ax3.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "time_vs_sources.png"), dpi=150)

plt.show()
