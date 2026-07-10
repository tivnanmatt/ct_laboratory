#!/usr/bin/env python3
"""One-time extraction of the 80-source / 3-rotation (240-view) system geometry.

Loads the pickled StaticCTProjector3D (requires a working ct_laboratory
install, e.g. inside a recon container) and saves only the plain src/dst ray
endpoint tensors so the benchmark can load them anywhere with
torch.load(weights_only=True) and no ct_laboratory pickle dependency.
"""
import argparse
from pathlib import Path

import torch

DEFAULT_BASE = "/workspace/research-ring/recon/recon_dev/20260623_sparse_eig_3d/recon/base_projector_instance.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-projector", default=DEFAULT_BASE)
    p.add_argument("--out", default=str(Path(__file__).parent / "geometry" / "geometry_80src_3rot_240view.pt"))
    args = p.parse_args()

    obj = torch.load(args.base_projector, map_location="cpu", weights_only=False)
    src = obj.src.detach().cpu().contiguous()
    dst = obj.dst.detach().cpu().contiguous()
    payload = {
        "src": src,
        "dst": dst,
        "n_rays": src.shape[0],
        "description": "80 sources x 3 rotations = 240 views, 12288 detector pixels/view; "
                       "extracted from base_projector_instance.pt (StaticCTProjector3D, native 250x250x20 @ 2mm)",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved {out} src={tuple(src.shape)} dst={tuple(dst.shape)}")


if __name__ == "__main__":
    main()
