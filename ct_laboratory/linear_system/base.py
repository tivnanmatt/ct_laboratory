import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class LinearSystem(torch.nn.Module, ABC):
    def __init__(self):
        """
        This is the base class for all linear systems.
        
        It inherits from torch.nn.Module and ABC (Abstract Base Class).
        All linear systems should inherit from this class and implement the abstract methods.
        """
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear system.
        
        parameters:
            x: torch.Tensor
                The input tensor to the linear system.
        returns:
            result: torch.Tensor
                The result of applying the linear system to the input tensor.
        """
        pass
    
    def forward_LinearSystem(self):
        return self
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear system.
        
        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the transpose of the linear system to the input tensor.
        """
        raise NotImplementedError
    
    def transpose_LinearSystem(self):
        raise NotImplementedError
    
    def conjugate(self, x: torch.Tensor):
        """
        This method implements the conjugate of the linear system.
        
        Default implementation: conj(A(x)) = A(conj(x))
        
        parameters:
            x: torch.Tensor
                The input tensor to the conjugate of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the conjugate of the linear system to the input tensor.
        """
        return torch.conj(self.forward(torch.conj(x)))
    
    def conjugate_LinearSystem(self):
        raise NotImplementedError
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the conjugate transpose of the linear system.
        
        Default implementation: conj(A^T(y)) = A^T(conj(y))
        
        parameters:
            y: torch.Tensor
                The input tensor to the conjugate transpose of the linear system.
        returns:
            result: torch.Tensor
                The result of applying the conjugate transpose of the linear system to the input tensor.
        """
        return torch.conj(self.transpose(torch.conj(y)))
    
    def conjugate_transpose_LinearSystem(self):
        raise NotImplementedError
    
    def sqrt_LinearSystem(self):
        raise NotImplementedError
    
    def inv_LinearSystem(self):
        raise NotImplementedError
    
    def logdet(self):
        raise NotImplementedError
    
    def det(self):
        raise NotImplementedError
    
    def _pseudoinverse_weighted_average(self, y: torch.Tensor):
        """
        This method implements the pseudoinverse of the linear system using the weighted average method.

        parameters:
            y: torch.Tensor of shape
                The input tensor to the pseudoinverse_weighted_average of the linear system.
        returns:
            x: torch.Tensor 
                The result of applying the pseudoinverse_weighted_average of the linear system to the input tensor.
        """
        
        numerator = self.conjugate_transpose(y)
        
        denominator = self.conjugate_transpose(torch.ones_like(y))

        x = numerator / (denominator + 1e-10)  # Avoid division by zero
        
        return x

    def _pseudoinverse_conjugate_gradient(self, b, max_iter=1000, tol=1e-6, beta=1e-3, verbose=False):
        """
        This method implements the pseudoinverse of the linear system using the conjugate gradient method.

        It solves the linear system (A^T A + beta * I) x = A^T b for x, where A is the linear system.

        parameters:
            b: torch.Tensor of shape
                The input tensor to which the pseudo inverse of the linear system should be applied.
            max_iter: int
                The maximum number of iterations to run the conjugate gradient method.
            tol: float
                The tolerance for the conjugate gradient method.
            beta: float
                The regularization strength for the conjugate gradient method.
        returns:
            x_est: torch.Tensor
                The result of applying the pseudoinverse_conjugate_gradient of the linear system to the input tensor.
        """
        ATb = self.conjugate_transpose(b)
        x_est = self._pseudoinverse_weighted_average(b)
        
        r = ATb - self.conjugate_transpose(self.forward(x_est)) - beta * x_est
        p = r.clone()
        rsold = torch.dot(r.flatten(), r.flatten())
        
        for i in range(max_iter):
            if verbose:
                print("Inverting ", self.__class__.__name__, " with conjugate_gradient. Iteration: {}, Residual: {}".format(i, torch.sqrt(torch.abs(rsold))))
            ATAp = self.conjugate_transpose(self.forward(p)) + beta * p
            alpha = rsold / torch.dot(p.flatten(), ATAp.flatten())
            x_est += alpha * p
            r -= alpha * ATAp
            rsnew = torch.dot(r.flatten(), r.flatten())
            if torch.sqrt(torch.abs(rsnew)) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x_est
    
    def pseudoinverse(self, y, method=None, **kwargs):
        """
        This method implements the pseudo inverse of the linear system.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear system should be applied.
            method: str
                The method to use for computing the pseudo inverse. If None, the method is chosen automatically.
            kwargs: dict
                Keyword arguments to be passed to the method.
        """

        if method is None:
            method = 'conjugate_gradient'

        assert method in ['weighted_average', 'conjugate_gradient'], "The method should be either 'weighted_average' or 'conjugate_gradient'."

        if method == 'weighted_average':
            return self._pseudoinverse_weighted_average(y)
        elif method == 'conjugate_gradient':
            return self._pseudoinverse_conjugate_gradient(y, **kwargs)
    
    def mat_add(self, M):
        raise NotImplementedError
    
    def mat_sub(self, M):
        raise NotImplementedError
    
    def mat_mul(self, M):
        raise NotImplementedError
    
    def __mul__(self, x):
        return self.forward(x)
    
    def __add__(self, M):
        return self.mat_add(M)
    
    def __sub__(self, M):
        return self.mat_sub(M)
    
    def __matmul__(self, M):
        from .composite import CompositeLinearSystem
        return CompositeLinearSystem([self, M]) 