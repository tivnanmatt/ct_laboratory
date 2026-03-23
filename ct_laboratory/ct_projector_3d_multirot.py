import os
import gc
import math

import numpy as np
import torch
from torch import nn
import time

from .ct_projector_3d_module import CTProjector3DModule


class MultiRotationProjector(nn.Module):
    """
    Rolling-window multi-rotation projector for helical-like CT scans.

    For each rotation the appropriate z-window is sliced from the full volume
    and projected through ``base_projector``.

    When ``base_projector`` is an ``OrderedSubsetProjector`` the subset-aware
    path is taken automatically: for each subset the cached weights are loaded
    once and applied to **all** rotations before moving to the next subset.
    This avoids repeated weight computation while keeping GPU memory bounded.

    Both forward and back_project return / accept a 2-D tensor of shape
    ``[n_rays, n_rotations]``.
    """

    def __init__(
        self,
        base_projector,
        n_rotations,
        shift_offset,
        n_z_window,
        scan_center_slice=None,
        padding_mode='repeat',
        reflection_padding_slices=None,
        verbose=False,
    ):
        super().__init__()
        self.base_projector = base_projector
        self.n_rotations = int(n_rotations)
        self.shift_offset = float(shift_offset)
        self.n_z_window = int(n_z_window)
        self.scan_center_slice = scan_center_slice
        self.padding_mode = padding_mode
        self.reflection_padding_slices = reflection_padding_slices
        self.verbose = verbose

    # ── geometry helpers ──────────────────────────────────────────────────────

    def _get_rotation_z_bounds(self, full_volume_z_size, rot_idx):
        """Return (start_z, end_z) in voxel coordinates for rot_idx."""
        if self.scan_center_slice is not None:
            center_vol = float(self.scan_center_slice) - 0.5
        else:
            center_vol = (full_volume_z_size - 1) / 2.0
        start_center = center_vol - (self.n_rotations - 1) / 2.0 * self.shift_offset
        rot_center   = start_center + rot_idx * self.shift_offset
        start_z = int(round(rot_center - (self.n_z_window - 1) / 2.0))
        return start_z, start_z + self.n_z_window

    def _extract_volume_slice(self, full_volume, start_z, end_z):
        """Extract z-window from full_volume, padding out-of-bounds edges."""
        vol_z   = full_volume.shape[-1]
        v_start = max(0, start_z)
        v_end   = min(vol_z, end_z)
        if v_start < v_end:
            v_slice = full_volume[..., v_start:v_end]
        else:
            v_slice = torch.zeros(
                (*full_volume.shape[:-1], 0),
                device=full_volume.device, dtype=full_volume.dtype,
            )
        pad_l = v_start - start_z
        pad_r = end_z - v_end
        if pad_l > 0 or pad_r > 0:
            if self.padding_mode == 'reflect':
                # torch.nn.functional.pad 'reflect' only supports 3D-5D tensors
                # and doesn't support 3D padding on the last dimension in the same way 
                # for very large pads (padding must be smaller than the dimension size).
                # Since we are just padding on the last dimension, we can do it manually.
                
                # We need to pad by pad_l on the left and pad_r on the right.
                # 'reflect' padding in torch: the first pixel is NOT repeated.
                # (e.g., [1, 2, 3] padded by 1 on each side becomes [2, 1, 2, 3, 2])
                # However, the user said "flip the volume in z direction to get the effective volume".
                
                # Let's use a simpler approach for CT: reflect at the boundary.
                # If we need many pixels of padding, we might need multiple flips,
                # but usually pad_l/pad_r are small.
                
                # Manual reflection padding for the last dimension:
                def get_padded_slice(vol, start, end):
                    z_dim = vol.shape[-1]
                    indices = torch.arange(start, end, device=vol.device)
                    # Reflect indices: 
                    # If idx < 0: -idx
                    # If idx >= z_dim: (z_dim - 1) - (idx - (z_dim - 1)) = 2*z_dim - 2 - idx
                    # This is reflection about the center of the edge voxel.
                    # Standard 'reflect' (like numpy or torch) reflects about the boundary.
                    
                    # Implementation of 'reflect' padding (boundary reflection):
                    # 0 1 2 | (edge) | 2 1 0  <- No, that's 'symmetric'
                    # 1 2 | 3 2 1 | 0  <- 'reflect'
                    
                    # Actually, let's use torch.nn.functional.pad if possible, 
                    # but it only supports certain pad sizes.
                    # Let's do a more robust index-based reflection.
                    
                    # Vectorized version:
                    curr_indices = indices.clone()
                    while True:
                        mask_neg = curr_indices < 0
                        mask_pos = curr_indices >= z_dim
                        if not (mask_neg.any() or mask_pos.any()):
                            break
                        curr_indices[mask_neg] = -curr_indices[mask_neg]
                        curr_indices[mask_pos] = 2 * z_dim - 2 - curr_indices[mask_pos]
                    
                    v_out = vol[..., curr_indices]

                    # Zero out regions beyond reflection limit if provided
                    if self.reflection_padding_slices is not None:
                        limit = self.reflection_padding_slices
                        mask_zero = (indices < -limit) | (indices >= z_dim + limit)
                        if mask_zero.any():
                            v_out[..., mask_zero] = 0.0

                    return v_out

                return get_padded_slice(full_volume, start_z, end_z).contiguous()
            elif self.padding_mode == 'repeat':
                # Map out-of-bounds indices to the nearest boundary (0 or vol_z-1)
                indices = torch.arange(start_z, end_z, device=full_volume.device)
                
                # Identify which slices are out-of-bounds and need to be detached
                is_out_of_bounds = (indices < 0) | (indices >= vol_z)
                
                clamped_indices = torch.clamp(indices, 0, vol_z - 1)
                v_slice = full_volume[..., clamped_indices]
                
                if is_out_of_bounds.any():
                    # We need to detach the padded region so it doesn't accumulate 
                    # gradients back to the boundary slice.
                    # Create a clone to avoid modifying full_volume in-place.
                    v_slice = v_slice.clone()
                    v_slice[..., is_out_of_bounds] = v_slice[..., is_out_of_bounds].detach()
                
                return v_slice.contiguous()
            else:
                v_slice = torch.nn.functional.pad(v_slice, (pad_l, pad_r), mode='constant', value=0.0)
        return v_slice.contiguous()

    # ── forward ───────────────────────────────────────────────────────────────

    def _forward_standard(self, full_volume):
        """Loop rotations → slice z-window → project. Returns [n_rays, n_rot]."""
        if not full_volume.is_contiguous():
            full_volume = full_volume.contiguous()
        z_size = full_volume.shape[-1]
        out = []
        for i in range(self.n_rotations):
            if self.verbose:
                print(f"  [Rot {i+1}/{self.n_rotations}] forward...", flush=True)
            start_z, end_z = self._get_rotation_z_bounds(z_size, i)
            v_slice = self._extract_volume_slice(full_volume, start_z, end_z)
            out.append(self.base_projector(v_slice))
        return torch.stack(out, dim=1).contiguous()  # [n_rays, n_rot]

    def _forward_subset(self, full_volume):
        """
        Subset-aware forward: per-subset weights loaded once, applied to all
        rotations. Returns [n_rays, n_rot].
        """
        if isinstance(full_volume, (list, tuple)):
            return self._forward_subset_multi(full_volume)
            
        if not full_volume.is_contiguous():
            full_volume = full_volume.contiguous()
        osp    = self.base_projector
        z_size = full_volume.shape[-1]
        device = full_volume.device
        sino_out = torch.zeros(
            osp.n_total_rays_core, self.n_rotations,
            device=device, dtype=full_volume.dtype,
        )
        for j in range(osp.n_subsets):
            start_ray = int(osp.subset_boundaries[j])
            end_ray   = int(osp.subset_boundaries[j + 1])
            if self.verbose:
                print(
                    f"  [Subset {j+1}/{osp.n_subsets}] rays {start_ray}-{end_ray} "
                    f"({end_ray-start_ray} rays)", flush=True,
                )
            subset_proj = osp._get_cached_subset_projector(j)
            subset_proj.to(device)
            for i in range(self.n_rotations):
                if self.verbose:
                    print(f"    → rotation {i+1}/{self.n_rotations}", flush=True)
                start_z, end_z = self._get_rotation_z_bounds(z_size, i)
                v_slice = self._extract_volume_slice(full_volume, start_z, end_z)
                sino_out[start_ray:end_ray, i] = subset_proj(v_slice)
            
            # Optimization: Only clear cache if we're actually memory constrained
            if osp.n_subsets > 1:
                del subset_proj
                gc.collect()
                torch.cuda.empty_cache()
            else:
                # If single subset, we might want to keep it on card
                pass
        return sino_out.contiguous()

    def _forward_subset_multi(self, volumes):
        """
        Multi-volume subset-aware forward: per-subset weights loaded once,
        applied to multiple volumes and all rotations.
        Returns list of [n_rays, n_rot] tensors.
        """
        osp = self.base_projector
        device = volumes[0].device
        z_size = volumes[0].shape[-1]
        
        sinos_out = [
            torch.zeros(osp.n_total_rays_core, self.n_rotations, device=device, dtype=v.dtype)
            for v in volumes
        ]
        
        for j in range(osp.n_subsets):
            start_ray = int(osp.subset_boundaries[j])
            end_ray   = int(osp.subset_boundaries[j + 1])
            if self.verbose:
                print(f"  [Subset {j+1}/{osp.n_subsets}] multi-volume projection...", flush=True)
                
            subset_proj = osp._get_cached_subset_projector(j)
            subset_proj.to(device)
            
            for i in range(self.n_rotations):
                if self.verbose and (i % 5 == 0 or i == self.n_rotations - 1):
                    print(f"    → rotations {i+1}-{min(i+5, self.n_rotations)}/{self.n_rotations}", flush=True)
                start_z, end_z = self._get_rotation_z_bounds(z_size, i)
                
                for v_idx, vol in enumerate(volumes):
                    v_slice = self._extract_volume_slice(vol, start_z, end_z)
                    sinos_out[v_idx][start_ray:end_ray, i] = subset_proj(v_slice)
            
            if osp.n_subsets > 1:
                del subset_proj
                gc.collect()
                torch.cuda.empty_cache()
                
        return [s.contiguous() for s in sinos_out]

    def forward(self, full_volume):
        """Forward-project full_volume → [n_rays, n_rotations]."""
        from .ct_projector_3d_subsets import SubsetProjector
        if isinstance(self.base_projector, SubsetProjector):
            return self._forward_subset(full_volume)
        return self._forward_standard(full_volume)

    # ── back-projection ───────────────────────────────────────────────────────

    def _back_project_standard(self, sino, full_volume_z_size):
        """sino: [n_rays, n_rot]. Returns [n_x, n_y, full_volume_z_size]."""
        bp     = self.base_projector
        device = sino.device
        # resolve core module for shape
        core = bp
        while hasattr(core, 'base_projector') and not isinstance(core, CTProjector3DModule):
            core = core.base_projector
        full_bp = torch.zeros(
            (core.n_x, core.n_y, full_volume_z_size),
            device=device, dtype=torch.float32,
        )
        for i in range(self.n_rotations):
            start_z, end_z = self._get_rotation_z_bounds(full_volume_z_size, i)
            bp_window = bp.back_project(sino[:, i].contiguous())
            
            if self.padding_mode == 'reflect':
                orig_indices = torch.arange(start_z, end_z, device=device)
                indices = orig_indices.clone()
                while True:
                    mask_neg = indices < 0
                    mask_pos = indices >= full_volume_z_size
                    if not (mask_neg.any() or mask_pos.any()):
                        break
                    indices[mask_neg] = -indices[mask_neg]
                    indices[mask_pos] = 2 * full_volume_z_size - 2 - indices[mask_pos]
                
                if self.reflection_padding_slices is not None:
                    limit = self.reflection_padding_slices
                    mask_valid = (orig_indices >= -limit) & (orig_indices < full_volume_z_size + limit)
                    if not mask_valid.all():
                        indices = indices[mask_valid]
                        bp_window = bp_window[..., mask_valid]
                
                # Scatter add across z indices.
                # However, bp_window is [n_x, n_y, n_z_window] and we want to 
                # accumulate into full_bp [n_x, n_y, full_volume_z_size]
                # over the last dimension based on indices.
                
                # Standard scatter_add works on specified dimension:
                # Need to use .index_add_ if we have many indices, 
                # but we're doing it slice-by-slice.
                
                # Actually, index_add_ is faster:
                full_bp.index_add_(2, indices, bp_window)
            elif self.padding_mode == 'repeat':
                # Map out-of-bounds indices to nearest boundary
                indices = torch.arange(start_z, end_z, device=device)
                indices = torch.clamp(indices, 0, full_volume_z_size - 1)
                full_bp.index_add_(2, indices, bp_window)
            else:
                v_start = max(0, start_z)
                v_end   = min(full_volume_z_size, end_z)
                p_start = v_start - start_z
                p_end   = p_start + (v_end - v_start)
                if v_start < v_end:
                    full_bp[..., v_start:v_end] += bp_window[..., p_start:p_end]
        return full_bp.contiguous()

    def _back_project_subset(self, sino, full_volume_z_size):
        """sino: [n_rays, n_rot]. Returns [n_x, n_y, full_volume_z_size]."""
        osp    = self.base_projector
        device = sino.device
        full_bp = torch.zeros(
            (osp.n_x, osp.n_y, full_volume_z_size),
            device=device, dtype=torch.float32,
        )
        for j in range(osp.n_subsets):
            start_ray = int(osp.subset_boundaries[j])
            end_ray   = int(osp.subset_boundaries[j + 1])
            if self.verbose:
                print(
                    f"  [Subset {j+1}/{osp.n_subsets}] rays {start_ray}-{end_ray} "
                    f"back-projecting...", flush=True,
                )
            subset_proj = osp._get_cached_subset_projector(j)
            subset_proj.to(device)
            for i in range(self.n_rotations):
                start_z, end_z = self._get_rotation_z_bounds(full_volume_z_size, i)
                bp_window = subset_proj.back_project(sino[start_ray:end_ray, i].contiguous())
                
                if self.padding_mode == 'reflect':
                    orig_indices = torch.arange(start_z, end_z, device=device)
                    indices = orig_indices.clone()
                    while True:
                        mask_neg = indices < 0
                        mask_pos = indices >= full_volume_z_size
                        if not (mask_neg.any() or mask_pos.any()):
                            break
                        indices[mask_neg] = -indices[mask_neg]
                        indices[mask_pos] = 2 * full_volume_z_size - 2 - indices[mask_pos]
                    
                    if self.reflection_padding_slices is not None:
                        limit = self.reflection_padding_slices
                        mask_valid = (orig_indices >= -limit) & (orig_indices < full_volume_z_size + limit)
                        if not mask_valid.all():
                            indices = indices[mask_valid]
                            bp_window = bp_window[..., mask_valid]
                    
                    full_bp.index_add_(2, indices, bp_window)
                elif self.padding_mode == 'repeat':
                    indices = torch.arange(start_z, end_z, device=device)
                    indices = torch.clamp(indices, 0, full_volume_z_size - 1)
                    full_bp.index_add_(2, indices, bp_window)
                else:
                    v_start = max(0, start_z)
                    v_end   = min(full_volume_z_size, end_z)
                    p_start = v_start - start_z
                    p_end   = p_start + (v_end - v_start)
                    if v_start < v_end:
                        full_bp[..., v_start:v_end] += bp_window[..., p_start:p_end]
            
            # Optimization: Only clear cache if we're actually memory constrained
            if osp.n_subsets > 1:
                del subset_proj
                gc.collect()
                torch.cuda.empty_cache()
            else:
                pass
        return full_bp.contiguous()

    def back_project(self, sino, full_volume_z_size):
        """
        Back-project sino ([n_rays, n_rotations]) into a volume of depth
        full_volume_z_size. Returns [n_x, n_y, full_volume_z_size].
        """
        from .ct_projector_3d_subsets import SubsetProjector
        if isinstance(self.base_projector, SubsetProjector):
            return self._back_project_subset(sino, full_volume_z_size)
        return self._back_project_standard(sino, full_volume_z_size)
