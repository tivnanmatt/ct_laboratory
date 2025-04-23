#!/usr/bin/env python
import math
import torch
import matplotlib.pyplot as plt
import os

# from ct_laboratory.staticct_projector_2d import UniformStaticCTProjector2D

current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "test_outputs")
os.makedirs(output_dir, exist_ok=True)

def build_circular_phantom(n_row, n_col, radius=40.0):
    phantom = torch.zeros(n_row, n_col, dtype=torch.float32)
    row_c = (n_row - 1) / 2.0
    col_c = (n_col - 1) / 2.0
    for r in range(n_row):
        for c in range(n_col):
            if (r - row_c)**2 + (c - col_c)**2 < radius**2:
                phantom[r, c] = 1.0
    return phantom

def main():
    n_row, n_col = 256, 256
    phantom = build_circular_phantom(n_row, n_col, radius=80.0)

    # Instead of pixel_size=1.0, define M,b for (row,col)->(x,y):
    # For simplicity, M=identity, b=0 => row= y, col= x
    M_2d = torch.eye(2, dtype=torch.float32)
    b_2d = torch.zeros(2, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phantom = phantom.to(device)

    # Example parameters
    n_source = 200       # number of sources around the circle
    n_frame = n_source   # one frame per source (i.e. only one source active per frame)
    n_module = 48       # number of detector modules
    det_n_col = 48      # detector columns per module
    det_spacing = 1.0   # detector spacing in mm
    source_radius = 380.00  # source-to-center distance (mm)
    module_radius = 366.70  # module-to-center distance (mm)

    # Build per-frame M_gantry, b_gantry => identity
    M_list = []
    b_list = []
    active_src = torch.zeros(n_frame, n_source, dtype=torch.bool)
    for i in range(n_frame):
        M_ = torch.eye(2, dtype=torch.float32)
        b_ = torch.zeros(2, dtype=torch.float32)
        b_[0] = n_col / 2.0
        b_[1] = n_row / 2.0
        M_list.append(M_)
        b_list.append(b_)
        # One source active per frame
        active_src[i, i] = True

    M_gantry = torch.stack(M_list, dim=0)
    b_gantry = torch.stack(b_list, dim=0)

    from ct_laboratory.staticct_projector_2d import UniformStaticCTProjector2D

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
        active_sources=active_src,
        backend="cuda"
    ).to(device)


    

    src = torch.reshape(projector.src, (n_frame, n_module*det_n_col, 2)).cpu()
    dst = torch.reshape(projector.dst, (n_frame, n_module*det_n_col, 2)).cpu()
    plt.figure()
    plt.axes(aspect='equal')
    plt.plot(src[0, 0, 0], src[0, 0, 1], 'o', color='b', label='source 0, module 0')
    plt.plot(dst[0, :, 0], dst[0, :, 1], 'o', color='r', label='detector 0, module 0')
    plt.savefig(os.path.join(output_dir, "uniform_static_2d_geometry.png"))


    with torch.no_grad():
        sinogram_1d = projector(phantom)

    # The shape is [n_frame, n_source, n_module, det_n_col] if all sources were active every frame.
    # But we have only 1 source active per frame => effectively shape is [n_frame, 1, n_module, det_n_col].
    # The total rays => n_frame * n_source * n_module * det_n_col => 8 * 8 * 4 * 40 => 10240
    # But actually only 8 * 1 * 4 * 40 => 1280 are used. We must confirm the code's generation matches that.
    # 
    # Because we used an identity "active_sources" pattern, each frame has exactly 1 source active => total rays:
    expected_N = n_frame * 1 * n_module * det_n_col  # 8*1*4*40=1280
    if sinogram_1d.numel() != expected_N:
        print(f"WARNING: Expected {expected_N} rays, got {sinogram_1d.numel()} => geometry mismatch?")

    # Reshape => [n_frame, n_module, det_n_col]


    sinogram = sinogram_1d.reshape(n_frame, n_module, det_n_col).cpu()
    # flip the det col dim
    # sinogram = torch.flip(sinogram, [2])
    # sinogram = sinogram.permute(0, 2, 1)  # [n_frame, det_n_col, n_module] => [n_frame, n_module, det_n_col]
    sinogram = sinogram.reshape(n_frame, n_module*det_n_col)  # [n_frame, n_module*det_n_col]
    sinogram = sinogram.permute(1, 0)  # [n_module*det_n_col, n_frame]

    # Suppose we treat frames = sources interchangeably => shape => [n_source, n_module, det_n_col]
    # We'll display the first source => sinogram[0].
    # Flatten the module dimension so we can show it as an "image" => shape [n_module, det_n_col].
    # For clarity, let's just show sinogram[0] as is => [n_module, det_n_col].
    # data_source0 = sinogram[0].cpu().numpy()  # shape => [n_module, det_n_col]

    # Let's display that as a 2D image => n_module along vertical, det_col along horizontal.
    plt.figure(figsize=(8,4))
    plt.imshow(sinogram, cmap='gray', origin='lower', aspect='auto')
    plt.title("2D Uniform Static Sinogram")
    plt.xlabel("View Index")
    plt.ylabel("Module/Column Index")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "uniform_static_2d_sinogram.png"))
    plt.close()

if __name__ == "__main__":
    main()
