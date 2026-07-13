import torch
from .base import LinearSystem


class RowSparseLinearSystem(LinearSystem):
    """
    This class implements a row sparse linear system that can be used in a PyTorch model.

    Row sparse linear systems have a small number of non-zero weights for each input element.
    """

    def __init__(self, input_shape, output_shape, indices, weights):
        """
        Initialize the row sparse linear system.

        Parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
            indices: torch.Tensor of shape [num_weights, *input_shape]
                The 1D indices of the flattened output tensor that each weight corresponds to.
            weights: torch.Tensor of shape [num_weights, *input_shape]
                The weights of the linear system.
        """
        super().__init__()

        # Check that indices and weights have the same shape.
        assert indices.shape == weights.shape, "Indices and weights must have the same shape."
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.indices = indices
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the forward pass of the linear system.

        Parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear system.
                
        Returns:
            torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear system to the input tensor.
        """
        batch_size, num_channel = x.shape[:2]
        assert x.shape[2:] == self.input_shape, "Input tensor shape doesn't match the specified input shape."

        result = torch.zeros(batch_size, num_channel, *self.output_shape, dtype=x.dtype, device=x.device)
        
        results_flattened = result.view(batch_size, num_channel, -1)
        x_flattened = x.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):
            for b in range(batch_size):
                for c in range(num_channel):
                    results_flattened[b, c].index_add_(0, self.indices[i].flatten(), 
                                                     (x_flattened[b, c] * self.weights[i].flatten()))
        return result.view(batch_size, num_channel, *self.output_shape)

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the transpose operation.

        Parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the transpose of the linear system.
                
        Returns:
            torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the transpose of the linear system to the input tensor.
        """
        batch_size, num_channel = y.shape[:2]
        
        assert y.shape[2:] == self.output_shape, "Input tensor shape to adjoint doesn't match the specified output_shape of the linear system."

        result = torch.zeros(batch_size, num_channel, *self.input_shape, dtype=y.dtype, device=y.device)
        result_flattened = result.view(batch_size, num_channel, -1)
        
        # Flatten the input tensor
        y_flattened = y.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):  # Loop over num_weights
            values = y_flattened[:, :, self.indices[i].flatten()]  # Adding an additional dimension for broadcasting
            result_flattened += self.weights[i].view(1, 1, -1) * values

        result = result_flattened.view(batch_size, num_channel, *self.input_shape)
        return result

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the conjugate operation.

        Parameters:
            x: torch.Tensor
                The input tensor.
                
        Returns:
            torch.Tensor: The conjugate of the input.
        """
        batch_size, num_channel = x.shape[:2]
        assert x.shape[2:] == self.input_shape, "Input tensor shape doesn't match the specified input shape."

        result = torch.zeros(batch_size, num_channel, *self.output_shape, dtype=x.dtype, device=x.device)
        
        results_flattened = result.view(batch_size, num_channel, -1)
        x_flattened = x.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):
            for b in range(batch_size):
                for c in range(num_channel):
                    results_flattened[b, c].index_add_(0, self.indices[i].flatten(), 
                                                     (x_flattened[b, c] * self.weights[i].conj().flatten()))
        return result.view(batch_size, num_channel, *self.output_shape)

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply the conjugate transpose operation.

        Parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the conjugate transpose of the linear system.
                
        Returns:
            torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the conjugate transpose of the linear system to the input tensor.
        """
        batch_size, num_channel = y.shape[:2]
        
        assert y.shape[2:] == self.output_shape, "Input tensor shape to adjoint doesn't match the specified output_shape of the linear system."

        result = torch.zeros(batch_size, num_channel, *self.input_shape, dtype=y.dtype, device=y.device)
        result_flattened = result.view(batch_size, num_channel, -1)
        
        # Flatten the input tensor
        y_flattened = y.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):  # Loop over num_weights
            values = y_flattened[:, :, self.indices[i].flatten()]  # Adding an additional dimension for broadcasting
            result_flattened += self.weights[i].conj().view(1, 1, -1) * values

        result = result_flattened.view(batch_size, num_channel, *self.input_shape)
        return result

    def mat_add(self, other):
        """Matrix addition with another RowSparseLinearSystem."""
        if not isinstance(other, RowSparseLinearSystem):
            raise NotImplementedError("Addition only supported for RowSparseLinearSystem.")
        
        if self.input_shape != other.input_shape or self.output_shape != other.output_shape:
            raise ValueError("Shape mismatch for addition.")
        
        # Add weights if indices are the same
        if torch.allclose(self.indices, other.indices):
            new_weights = self.weights + other.weights
            return RowSparseLinearSystem(self.input_shape, self.output_shape, self.indices, new_weights)
        else:
            # For different indices, we need to merge them
            # This is a simplified implementation - in practice, you might want more sophisticated merging
            raise NotImplementedError("Addition with different indices not yet implemented.")

    def mat_sub(self, other):
        """Matrix subtraction with another RowSparseLinearSystem."""
        if not isinstance(other, RowSparseLinearSystem):
            raise NotImplementedError("Subtraction only supported for RowSparseLinearSystem.")
        
        if self.input_shape != other.input_shape or self.output_shape != other.output_shape:
            raise ValueError("Shape mismatch for subtraction.")
        
        # Subtract weights if indices are the same
        if torch.allclose(self.indices, other.indices):
            new_weights = self.weights - other.weights
            return RowSparseLinearSystem(self.input_shape, self.output_shape, self.indices, new_weights)
        else:
            raise NotImplementedError("Subtraction with different indices not yet implemented.")

    def mat_mul(self, other):
        """Matrix multiplication with another linear system or tensor."""
        if isinstance(other, torch.Tensor):
            return self.forward(other)
        else:
            raise NotImplementedError("Matrix multiplication with other operators not yet implemented.") 