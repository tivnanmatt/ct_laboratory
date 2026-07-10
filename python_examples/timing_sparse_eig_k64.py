#!/usr/bin/env python
"""Timing test: SparseEigenDecomposition(k=64) on the standard StaticCT 3D projector config."""
import time
import torch

def main():
    n_x, n_y, n_z = 128, 128, 16

    M_3d = torch.eye(3, dtype=torch.float32)
    b_3d = torch.zeros(3, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    n_source = 200
    n_frame = n_source
    n_module = 48
    modules_per_source = n_module // n_source if n_module > n_source else 1
    det_nx_per_module = 48
    det_ny_per_module = 16
    det_spacing_x = 1.0
    det_spacing_y = 2.0
    source_radius = 380.00
    module_radius = 366.70
    source_z_offset = 0.0
    module_z_offset = 0.0

    M_list, b_list = [], []
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
    from ct_laboratory.sparse_eigen_decomposition import SparseEigenDecomposition

    t_build0 = time.time()
    projector = UniformStaticCTProjector3D(
        n_x=n_x, n_y=n_y, n_z=n_z,
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
        modules_per_source=modules_per_source,
        M=M_3d, b=b_3d,
        backend="cuda", device=device,
    ).to(device)
    print(f"projector built in {time.time()-t_build0:.2f}s  volume=({n_x},{n_y},{n_z})  N={n_x*n_y*n_z}")

    dec = SparseEigenDecomposition(projector, k=64)

    t0 = time.time()
    dec.compute_weights(n_iters=30, verbose=True)
    elapsed = time.time() - t0
    print(f"\n=== k=64 compute_weights: {elapsed:.2f}s ({elapsed/60:.2f} min) ===")

if __name__ == "__main__":
    main()
