import torch
from .base import LinearSystem

class SquareLinearSystem(LinearSystem):
    def __init__(self):
        super().__init__()

    def input_shape_given_output_shape(self, output_shape):
        """
        This method computes the input shape given the output shape.
        
        For square operators, input_shape = output_shape.
        
        parameters:
            output_shape: tuple
                The shape of the output tensor.
        returns:
            input_shape: tuple
                The shape of the input tensor.
        """
        return output_shape 