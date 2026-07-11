import torch
from .sparse_col import ColSparseLinearSystem


class NearestNeighborInterpolator(ColSparseLinearSystem):
    """
    This class implements a nearest neighbor interpolator that can be used in a PyTorch model.
    
    It inherits from the ColSparseLinearSystem class.
    
    The weights are precomputed using nearest neighbor interpolation.
    """
    
    def __init__(self, num_row, num_col, interp_points):
        """
        Initialize the nearest neighbor interpolator.
        
        Parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            interp_points: torch.Tensor of shape [num_points, 2]
                The coordinates of the interpolation points.
        """
        
        # Store the number of rows and columns
        self.num_row = num_row
        self.num_col = num_col
        
        # Extract x and y coordinates of interp_points
        ix = interp_points[:, 0]
        iy = interp_points[:, 1]
        
        # Calculate the nearest indices for interpolation
        self.ix0 = torch.round(ix).long()
        self.iy0 = torch.round(iy).long()
        
        # mask for out-of-boundary indices
        self.oob_mask = (self.ix0 >= self.num_row) | (self.iy0 >= self.num_col) | (self.ix0 < 0) | (self.iy0 < 0)
        self.ix0[self.oob_mask] = 0
        self.iy0[self.oob_mask] = 0
        
        # Calculate weights for each value
        self.wa = torch.ones_like(self.ix0).type(torch.float32)
        self.wa[self.oob_mask] = 0
        
        indices = self._2d_to_1d_indices(self.ix0, self.iy0).view(1, -1)
        weights = self.wa.flatten().view(1, -1)
        
        super().__init__((num_row, num_col), (interp_points.shape[0],), indices, weights)
        
    def _2d_to_1d_indices(self, ix, iy):
        """Convert 2D indices to 1D indices."""
        return ix * self.num_col + iy
    
    def _1d_to_2d_indices(self, index):
        """Convert 1D indices to 2D indices."""
        ix = index // self.num_col
        iy = index % self.num_col
        return ix, iy 