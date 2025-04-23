#!/usr/bin/env python
import math
import torch
import matplotlib.pyplot as plt
import os

# from ct_laboratory.staticct_projector_3d import UniformStaticCTProjector3D

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

def build_3d_sphere(n_x, n_y, n_z, radius=20.0):
    vol = torch.zeros(n_x, n_y, n_z, dtype=torch.float32)
    cx = (n_x - 1)*0.5
    cy = (n_y - 1)*0.5
    cz = (n_z - 1)*0.5
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
    n_x, n_y, n_z = 128, 128, 16
    phantom = build_3d_sphere(n_x, n_y, n_z, radius=30.0)

    # M,b for (i,j,k)->(x,y,z)
    M_3d = torch.eye(3, dtype=torch.float32)
    b_3d = torch.zeros(3, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom = phantom.to(device)

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

    M_list = []
    b_list = []
    active_src = torch.zeros(n_frame, n_source, dtype=torch.bool)
    for i in range(n_frame):
        M__ = torch.eye(3, dtype=torch.float32)
        b__ = torch.zeros(3, dtype=torch.float32)
        b__[0] = n_x / 2
        b__[1] = n_y / 2
        b__[2] = n_z / 2

        M_list.append(M__)
        b_list.append(b__)
        active_src[i, i] = True

    M_gantry = torch.stack(M_list, dim=0)
    b_gantry = torch.stack(b_list, dim=0)

    from ct_laboratory.staticct_projector_3d import UniformStaticCTProjector3D

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

    src = torch.reshape(projector.src, (n_source, n_module*det_nx_per_module, det_ny_per_module, 3)).cpu()
    dst = torch.reshape(projector.dst, (n_source, n_module*det_nx_per_module, det_ny_per_module, 3)).cpu()

    plt.figure()
    plt.axes(aspect='equal')
    plt.plot(src[0, 0, 0, 0], src[0, 0, 0, 1], 'o', color='b')
    plt.plot(dst[0, :, 0, 0], dst[0, :, 0, 1], 'o', color='r')
    plt.savefig(os.path.join(output_dir, "uniform_static_3d_geometry_topview.png"))


    plt.figure()
    plt.axes(aspect='equal')
    plt.plot(src[0, 0, 0, 0], src[0, 0, 0, 2], 'o', color='b')
    plt.plot(dst[0, :, 0, 0], dst[0, :, 0, 2], 'o', color='r')
    plt.savefig(os.path.join(output_dir, "uniform_static_3d_geometry_sideview.png"))


    with torch.no_grad():
        sinogram_1d = projector(phantom)
    # For display, we can reshape to [n_frame, n_module, det_nx_per_module, det_ny_per_module].
    # Because we used exactly 1 source active per frame => shape is [n_source, n_module, 48, 48].
    # Then we can pick the first frame [0] for visualization
    sinogram_4d = sinogram_1d.view(n_source, n_module, det_nx_per_module, det_ny_per_module)


    # ------------------------
    # 4) Display
    # ------------------------

    # Show phantom mid-slice along Z
    mid_z = n_z // 2
    phantom_slice = phantom[:, :, mid_z].cpu().numpy()

    # Show sinogram for the 0th frame
    # shape => [n_module, 48, 48]. Let's just show sinogram_4d[0,0] => shape [48,48]
    # or you could mosaic them, but let's just pick the first module for demonstration
    sino_2d = sinogram_4d.reshape(n_source, n_module*det_nx_per_module, det_ny_per_module)[0].cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10,5))
    axs[0].imshow(phantom_slice, cmap="gray", origin="lower")
    axs[0].set_title(f"3D Phantom (z={mid_z} slice)")
    axs[1].imshow(sino_2d.transpose(), cmap="gray", origin="lower", aspect="auto")
    axs[1].set_title(f"Static 3D Sinogram (frame=0)")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "staticct_3d_sinogram.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved figure => {save_path}")

    plt.show()


    
if __name__ == "__main__":

    main()