
import torch
import torch.nn as nn

class QuadraticRegularizer3D(nn.Module):
    """
    3D Quadratic Smoothness Penalty. 
    
    Computes a sum of squared differences between neighboring voxels
    along x, y, and z directions (akin to a discrete 3D Laplacian).
    """
    def __init__(self):
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (n_x, n_y, n_z) or (batch, n_x, n_y, n_z).
        For simplicity assume shape is (n_x, n_y, n_z). If batch is used, 
        adapt the shift dims accordingly.
        """
        # Differences in x-direction
        diff_x = volume - torch.roll(volume, shifts=-1, dims=0)
        # Differences in y-direction
        diff_y = volume - torch.roll(volume, shifts=-1, dims=1)
        # Differences in z-direction
        diff_z = volume - torch.roll(volume, shifts=-1, dims=2)

        # Sum of squared differences
        reg_val = (diff_x ** 2).mean() + (diff_y ** 2).mean() + (diff_z ** 2).mean()

        return reg_val

class TVRegularizer3D(nn.Module):
    """
    3D Total Variation Penalty.
    
    Computes the sum of absolute differences between neighboring voxels
    along x, y, and z directions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (n_x, n_y, n_z) or (batch, n_x, n_y, n_z).
        For simplicity assume shape is (n_x, n_y, n_z).
        """
        # Differences in x-direction
        diff_x = volume - torch.roll(volume, shifts=-1, dims=0)
        # Differences in y-direction
        diff_y = volume - torch.roll(volume, shifts=-1, dims=1)
        # Differences in z-direction
        diff_z = volume - torch.roll(volume, shifts=-1, dims=2)

        # TV penalty = sum of absolute differences
        tv_val = diff_x.abs().mean() + diff_y.abs().mean() + diff_z.abs().mean()

        # tv_val = (diff_x ** 2) + (diff_y ** 2) + (diff_z ** 2)
        # tv_val = torch.sqrt(tv_val).mean()

        return tv_val
    