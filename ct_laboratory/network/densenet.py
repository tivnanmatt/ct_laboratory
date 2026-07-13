
import torch
import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_channels_list, activation='relu'):
        """
        Initializes the DenseNet model with the specified parameters.

        Args:
            input_shape (tuple): Shape of the input tensor.
            output_shape (tuple): Shape of the output tensor.
            hidden_channels_list (list): List of hidden layer sizes.
            activation (str or list): Activation function to use or list of activation functions.
        """
        super(DenseNet, self).__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        # Flatten the input shape
        self.input_size = int(torch.prod(torch.tensor(input_shape)))
        self.output_size = int(torch.prod(torch.tensor(output_shape)))

        layers = []

        # Ensure activation is a list and matches the length of hidden_channels_list
        if isinstance(activation, str):
            activation = [activation] * len(hidden_channels_list)
        if isinstance(activation, nn.Module):
            activation = [activation] * len(hidden_channels_list)
        elif isinstance(activation, list) and len(activation) != len(hidden_channels_list):
            raise ValueError("Length of activation functions list must match the length of hidden_channels_list")

        # Add the hidden layers with batch normalization and activation functions
        in_size = self.input_size
        for out_size, act in zip(hidden_channels_list, activation):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.BatchNorm1d(out_size))
            layers.append(self.get_activation(act))
            in_size = out_size

        # Add the output layer without batch normalization and activation
        layers.append(nn.Linear(in_size, self.output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the DenseNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        assert list(x.shape[-len(self.input_shape):]) == self.input_shape, f"Last dimensions of input tensor {x.shape} do not match the input shape {self.input_shape}"

        batch_shape = x.shape[:-len(self.input_shape)]
        batch_size = int(torch.prod(torch.tensor(batch_shape)))

        x = x.view(batch_size, -1)  # Flatten the input
        x = self.model(x)
        x = x.view(*batch_shape, *self.output_shape)  # Reshape the output
        return x

    def get_activation(self, activation):
        """
        Returns the activation function based on the provided name.

        Args:
            activation (str): Name of the activation function.

        Returns:
            nn.Module: Activation function.
        """
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'prelu':
            return nn.PReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'linear':
            return nn.Identity()
        elif isinstance(activation, nn.Module):
            return activation
        else:
            raise ValueError(f"Unsupported activation function: {activation}")