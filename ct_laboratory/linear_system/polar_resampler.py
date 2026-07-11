import torch
import torch.nn.functional as F
from .real import RealLinearSystem
from .interpolator_nearest import NearestNeighborInterpolator
from .interpolator_bilinear import BilinearInterpolator
from .interpolator_lanczos import LanczosInterpolator


class PolarCoordinateResampler(RealLinearSystem):
    """
    This class implements a polar coordinate transformation that can be used in a PyTorch model.
    """

    def __init__(self, num_row, num_col, theta_values, radius_values, row_center=None, col_center=None, interpolator=None):
        """
        Initialize the polar coordinate resampler.

        Parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            theta_values: torch.Tensor of shape [num_theta]
                The theta values, in radians, of the polar grid.
            radius_values: torch.Tensor of shape [num_radius]
                The radius values, in pixels, of the polar grid.
            row_center: int, optional
                The row center of the image. If None, calculated as (num_row - 1) // 2.
            col_center: int, optional
                The column center of the image. If None, calculated as (num_col - 1) // 2.
            interpolator: str, optional
                The interpolation method to use. Options: 'nearest', 'bilinear', 'lanczos'. Default: 'lanczos'.
        """
        super().__init__()

        # Store the theta and radius values
        self.theta_values = theta_values
        self.radius_values = radius_values
        self.num_row = num_row
        self.num_col = num_col

        # Store the number of theta and radius values
        self.num_theta = len(theta_values)
        self.num_radius = len(radius_values)

        # Calculate the center of the image so that it works for even and odd dimensions
        if row_center is None:
            row_center = (self.num_row - 1) // 2
        if col_center is None:
            col_center = (self.num_col - 1) // 2

        self.row_center = row_center
        self.col_center = col_center

        # Create a meshgrid of theta and radius values
        theta_mesh, radius_mesh = torch.meshgrid(theta_values, radius_values, indexing='ij')

        # Convert meshgrid to row, col coordinates
        #   where col is the horizontal axis, increasing from left to right
        #   and row is the vertical axis, increasing from top to bottom
        x_mesh = radius_mesh * torch.cos(theta_mesh)
        y_mesh = radius_mesh * torch.sin(theta_mesh)
        row_mesh = row_center - y_mesh + 1
        col_mesh = col_center + x_mesh + 1

        # Flatten and stack to get interp_points with shape [num_points, 2]
        interp_points = torch.stack((row_mesh.flatten(), col_mesh.flatten()), dim=1)
        
        if interpolator is None:
            interpolator = 'lanczos'
        
        assert interpolator in ['nearest', 'bilinear', 'lanczos'], "The interpolator must be one of 'nearest', 'bilinear', or 'lanczos'."

        if interpolator == 'nearest':
            self.interpolator = NearestNeighborInterpolator(self.num_row, self.num_col, interp_points)
        elif interpolator == 'bilinear':
            self.interpolator = BilinearInterpolator(self.num_row, self.num_col, interp_points)
        elif interpolator == 'lanczos':
            self.interpolator = LanczosInterpolator(self.num_row, self.num_col, interp_points, kernel_size=5)

        # Store the interp points
        self.interp_points = interp_points

        # Store shape for reshaping in forward method
        self.theta_mesh = theta_mesh
        self.radius_mesh = radius_mesh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward pass of the polar coordinate transformation.

        Parameters:
            x: torch.Tensor 
                The input image to which the polar coordinate transformation should be applied.
                
        Returns:
            torch.Tensor 
                The result of applying the polar coordinate transformation to the input image.
        """
        # Interpolate the values
        interpolated = self.interpolator(x)
        
        # Reshape to the original theta, r grid
        result = interpolated.view(*interpolated.shape[:2], *self.theta_mesh.shape)
        
        return result
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose pass of the polar coordinate transformation.

        Parameters:
            y: torch.Tensor 
                The input image to which the adjoint polar coordinate transformation should be applied.
                
        Returns:
            torch.Tensor 
                The result of applying the adjoint polar coordinate transformation to the input image.
        """
        # Flatten the polar image
        flattened = y.view(*y.shape[:2], -1)
        
        # Use the adjoint method of the interpolator
        result = self.interpolator.transpose(flattened)
        
        return result

    def mat_add(self, other):
        """Matrix addition with another PolarCoordinateResampler."""
        if not isinstance(other, PolarCoordinateResampler):
            raise NotImplementedError("Addition only supported for PolarCoordinateResampler.")
        
        # For polar resamplers, we can only add if they have the same configuration
        if (self.num_row != other.num_row or self.num_col != other.num_col or
            not torch.allclose(self.theta_values, other.theta_values) or
            not torch.allclose(self.radius_values, other.radius_values)):
            raise ValueError("Polar resamplers must have identical configuration for addition.")
        
        # Create a new resampler with the same configuration
        return PolarCoordinateResampler(
            self.num_row, self.num_col, self.theta_values, self.radius_values,
            self.row_center, self.col_center, 'lanczos'  # Default interpolator
        )

    def mat_sub(self, other):
        """Matrix subtraction with another PolarCoordinateResampler."""
        if not isinstance(other, PolarCoordinateResampler):
            raise NotImplementedError("Subtraction only supported for PolarCoordinateResampler.")
        
        # For polar resamplers, we can only subtract if they have the same configuration
        if (self.num_row != other.num_row or self.num_col != other.num_col or
            not torch.allclose(self.theta_values, other.theta_values) or
            not torch.allclose(self.radius_values, other.radius_values)):
            raise ValueError("Polar resamplers must have identical configuration for subtraction.")
        
        # Create a new resampler with the same configuration
        return PolarCoordinateResampler(
            self.num_row, self.num_col, self.theta_values, self.radius_values,
            self.row_center, self.col_center, 'lanczos'  # Default interpolator
        )

    def mat_mul(self, other):
        """Matrix multiplication with another linear operator or tensor."""
        if isinstance(other, torch.Tensor):
            return self.forward(other)
        else:
            raise NotImplementedError("Matrix multiplication with other operators not yet implemented.") 