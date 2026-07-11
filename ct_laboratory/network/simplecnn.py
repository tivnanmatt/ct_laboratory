import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels_list, activation, dim=2):
        """
        Simple CNN model for image reconstruction.

        Args:
            input_channels (int): Number of input channels.
            output_channels (int): Number of output channels.
            hidden_channels_list (list): List of hidden channels.
            activation (str or torch.nn.Module or list): Activation function(s) for hidden layers.
        """
        super(SimpleCNN, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        layers = []

        # Ensure activation is a list and matches the length of hidden_channels_list
        if isinstance(activation, str):
            if activation == 'relu':
                activation = nn.ReLU()
            elif activation == 'leakyrelu':
                activation = nn.LeakyReLU()
            elif activation == 'prelu':
                activation = nn.PReLU()
            elif activation == 'elu':
                activation = nn.ELU()
            elif activation == 'selu':
                activation = nn.SELU()
            elif activation == 'tanh':
                activation = nn.Tanh()
            elif activation == 'sigmoid':
                activation = nn.Sigmoid()
            elif activation == 'softmax':
                activation = nn.Softmax(dim=1)
            elif activation == 'silu':
                activation = nn.SiLU()
        if isinstance(activation, nn.Module):
            activation = [activation] * len(hidden_channels_list)
        elif isinstance(activation, list) and len(activation) != len(hidden_channels_list):
            raise ValueError("Length of activation functions list must match the length of hidden_channels_list")

        if dim == 2:
            Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3:
            Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else:
            raise ValueError("Invalid dim value. Must be 2 or 3.")
        self.dim = dim

        # Add the hidden layers with batch normalization and activation functions
        in_channels = self.input_channels
        for out_channels, act in zip(hidden_channels_list, activation):
            layers.append(Conv(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(BatchNorm(out_channels))
            layers.append(act)
            in_channels = out_channels

        # Add the output layer without batch normalization and activation (dim-aware)
        layers.append(Conv(in_channels, self.output_channels, kernel_size=3, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)