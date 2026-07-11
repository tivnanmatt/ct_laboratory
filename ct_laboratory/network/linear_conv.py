import torch
import torch.nn as nn

class LinearConv(nn.Module):
    """
    Linear Convolution layer with no activation function and no bias.
    Implements a single trainable filter.
    """
    
    def __init__(self, in_channels=1, out_channels=1, kernel_size=7, padding=None):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Set padding to maintain input size if not specified
        if padding is None:
            padding = kernel_size // 2
        
        # Single convolution layer with no bias
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False  # No bias term
        )
    
    def forward(self, x):
        """
        Forward pass - just applies the convolution without any activation.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        return self.conv(x) 