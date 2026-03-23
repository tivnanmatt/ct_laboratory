from torch import nn
import torch

from .quadratic_smoothness_prior import laplacian_operator_3d, laplacian_operator_2d


class TotalVariancePrior2D(nn.Module):
    """
    3D Total Variation Penalty.
    
    Computes the sum of absolute differences between neighboring voxels
    along x, y, and z directions.
    """
    def __init__(self, regularization_weight=1.0):
        self.regularization_weight = regularization_weight
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (n_x, n_y) or (batch, n_x, n_y).
        """
        # Differences in x-direction
        diff_x = volume - torch.roll(volume, shifts=-1, dims=0)
        # Differences in y-direction
        diff_y = volume - torch.roll(volume, shifts=-1, dims=1)

        # TV penalty = mean of absolute differences
        log_prior = -1.0 * diff_x.abs().mean() - 1.0 * diff_y.abs().mean()
        log_prior = self.regularization_weight * log_prior
        
        return log_prior

class TotalVariancePrior3D(nn.Module):
    """
    3D Total Variation Penalty.
    
    Computes the sum of absolute differences between neighboring voxels
    along x, y, and z directions.
    """
    def __init__(self, regularization_weight=1.0):
        self.regularization_weight = regularization_weight
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (n_x, n_y, n_z) or (batch, n_x, n_y, n_z).
        For simplicity assume shape is (n_x, n_y, n_z).
        """
        if self.regularization_weight == 0:
            return torch.tensor(0.0, device=volume.device, dtype=volume.dtype)

        # TV penalty = sum of absolute differences
        # Process dimensions sequentially to save memory
        log_prior = torch.tensor(0.0, device=volume.device, dtype=volume.dtype)
        for dim in range(volume.ndim):
            diff = volume - torch.roll(volume, shifts=-1, dims=dim)
            log_prior = log_prior - diff.abs().mean()
            del diff
            
        log_prior = self.regularization_weight * log_prior
        
        return log_prior
        # tv_val = (diff_x ** 2) + (diff_y ** 2) + (diff_z ** 2)
        # tv_val = torch.sqrt(tv_val).mean()

        # return tv_val
    
        # laplacian_response = laplacian_operator_3d(volume)
        # log_prior = -0.5 * self.regularization_weight * torch.sum(torch.sqrt(1e-2 + laplacian_response * volume))
        # # log_prior = -0.5 * self.regularization_weight * torch.sum(laplacian_response * volume)
        
        return log_prior