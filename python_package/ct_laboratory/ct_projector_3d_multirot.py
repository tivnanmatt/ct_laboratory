import torch
from torch import nn


class MultiRotationProjector(nn.Module):
    def __init__(self, projector, n_rot, shift_per_rot, shift_offset):
        super().__init__()
        self.projector = projector
        self.n_rot = n_rot
        self.shift_per_rot = shift_per_rot
        self.shift_offset = shift_offset

    def shift_volume(self, volume: torch.Tensor, shift_z: int) -> torch.Tensor:
        # """Shift the volume along z by 'shift_z' voxels using torch.roll."""
        return torch.roll(volume, shifts=shift_z, dims=(2,))

        # # pad with zeros then roll then take the center
        # # pad the volume
        # tmp_volume = torch.cat(
        #     [torch.zeros_like(volume[:, :, :shift_z]), volume, torch.zeros_like(volume[:, :, :shift_z])], dim=2
        # )

        # # roll the volume
        # tmp_volume = torch.roll(tmp_volume, shifts=shift_z, dims=(2,))

        # # take the center
        # return tmp_volume[:, :, shift_z:shift_z + volume.shape[2]].contiguous()

    def forward_project(self, volume):
        """Perform forward projection for multiple rotations (helical motion)."""
        # volume = volume
        sino_list = []
        for i in range(self.n_rot):
            vol_shifted = self.shift_volume(volume, i * self.shift_per_rot + self.shift_offset)
            sino_i = self.projector(vol_shifted)
            sino_list.append(sino_i)
        return torch.cat(sino_list, dim=0).view(-1)
    

    def back_project(self, sino):
        proj_per_rot = sino.shape[0] // self.n_rot
        for i in range(self.n_rot):
            volume_shifted = self.projector.back_project(sino[i * proj_per_rot:(i + 1) * proj_per_rot])
            if i == 0:
                volume = volume_shifted.clone()
            else:
                volume += self.shift_volume(volume_shifted, -i * self.shift_per_rot - self.shift_offset)    
        return volume

    def forward(self, x_atten):
        return self.forward_project(x_atten)