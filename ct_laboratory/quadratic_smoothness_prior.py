
import torch
import torch.nn as nn



# Define the quadratic smoothness regularization term
def laplacian_operator_3d(volume):
    # Compute the Laplacian using finite differences
    laplacian_response = torch.zeros_like(volume)
    laplacian_response -= torch.roll(volume, shifts=1, dims=0)
    laplacian_response -= torch.roll(volume, shifts=-1, dims=0)
    laplacian_response -= torch.roll(volume, shifts=1, dims=1)
    laplacian_response -= torch.roll(volume, shifts=-1, dims=1)
    laplacian_response -= torch.roll(volume, shifts=1, dims=2)
    laplacian_response -= torch.roll(volume, shifts=-1, dims=2)
    laplacian_response += 6 * volume
    return laplacian_response


# Define the quadratic smoothness regularization term
def laplacian_operator_2d(volume):
    # Compute the Laplacian using finite differences
    laplacian_response = torch.zeros_like(volume)
    laplacian_response -= torch.roll(volume, shifts=1, dims=0)
    laplacian_response -= torch.roll(volume, shifts=-1, dims=0)
    laplacian_response -= torch.roll(volume, shifts=1, dims=1)
    laplacian_response -= torch.roll(volume, shifts=-1, dims=1)
    laplacian_response += 4 * volume
    return laplacian_response

class QuadraticSmoothnessLogPrior3D(nn.Module):
    """
    3D Quadratic Smoothness Penalty. 
    
    Computes a sum of squared differences between neighboring voxels
    along x, y, and z directions (akin to a discrete 3D Laplacian).
    """
    def __init__(self, regularization_weight=1.0):
        self.regularization_weight = regularization_weight
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (n_x, n_y, n_z) or (batch, n_x, n_y, n_z).
        For simplicity assume shape is (n_x, n_y, n_z). If batch is used, 
        adapt the shift dims accordingly.
        """
        if self.regularization_weight == 0:
            return torch.tensor(0.0, device=volume.device, dtype=volume.dtype)

        # Using difference-based calculation for better numerical stability with large offsets (e.g. HU)
        # Process each dimension sequentially to save memory
        sum_sq_diff = 0
        for dim in range(volume.ndim):
            diff = volume - torch.roll(volume, shifts=-1, dims=dim)
            sum_sq_diff = sum_sq_diff + torch.sum(diff**2)
            del diff
        
        log_prior = - 0.5 * self.regularization_weight * sum_sq_diff
        
        return log_prior


class QuadraticSmoothnessLogPrior2D(nn.Module):
    """
    3D Quadratic Smoothness Penalty. 
    
    Computes a sum of squared differences between neighboring voxels
    along x, y, and z directions (akin to a discrete 3D Laplacian).
    """
    def __init__(self, regularization_weight=1.0):
        self.regularization_weight = regularization_weight
        super().__init__()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        volume: (n_x, n_y, n_z) or (batch, n_x, n_y, n_z).
        For simplicity assume shape is (n_x, n_y, n_z). If batch is used, 
        adapt the shift dims accordingly.
        """

        # Using difference-based calculation for better numerical stability with large offsets (e.g. HU)
        diff_x = volume - torch.roll(volume, shifts=-1, dims=0)
        diff_y = volume - torch.roll(volume, shifts=-1, dims=1)
        
        log_prior = - 0.5 * self.regularization_weight * torch.sum(diff_x**2 + diff_y**2)
        
        return log_prior
