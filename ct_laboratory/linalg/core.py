# random_tensor_laboratory/linalg/core.py
import torch

class LinearSystem(torch.nn.Module):
    def __init__(self):
        """
        This is an abstract class for linear operators.

        It inherits from torch.nn.Module.
        
        It requires the methods forward and transpose to be implemented.

        parameters:
            None
        """

        super(LinearSystem, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        return NotImplementedError
    
    def forward_LinearSystem(self):
        return self
    
    def transpose(self,y: torch.Tensor) -> torch.Tensor:
        """
        This method returns the transpose of the linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear operator.

        returns:
            z: torch.Tensor
                The transpose of the linear operator applied to y
        """
        
        # by default, use a forward pass applied to zeros and then transpose
        # this is not the most efficient way to do this, but it is the most general

        # check if self.input_shape is defined
        if hasattr(self, 'input_shape'):
            input_shape = self.input_shape
        else:
            if hasattr(self, 'input_shape_given_output_shape'):
                self.input_shape = self.input_shape_given_output_shape(y.shape)
            else:
                raise NotImplementedError("Subclass of rtl.linear_systems.LinearSystem must either define the input_shape, define the input_shape_given_output_shape(output_shape) method, or attribute or implement the transpose method.")

        _input = torch.zeros(input_shape, dtype=y.dtype, device=y.device)
        _input.requires_grad = True
        _output = self.forward(_input)
        # now use autograd to compute the transpose applied to y
        # we also need the transpose operation itself to be differentiable
        _output.backward(y, create_graph=True)
        # backpropagtion through the forward pass is the transpose pass
        z = _input.grad
        return z
    
    def transpose_LinearSystem(self):
        return TransposeLinearSystem(self)
    
    def conjugate(self, x: torch.Tensor):
        """
        This method returns the conjugate of the linear operator.

        parameters:
            x: torch.Tensor 
                The input tensor to the conjugate of the linear operator.

        returns:
            conjugate: LinearSystem object
                The conjugate of the linear operator.

        """
        return torch.conj(self.forward(torch.conj(x)))

    def conjugate_LinearSystem(self):
        return ConjugateLinearSystem(self)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            z: torch.Tensor 
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        z = torch.conj(self.transpose(torch.conj(y)))
        return z
    
    def conjugate_transpose_LinearSystem(self):
        return ConjugateTransposeLinearSystem(self)    
    
    def sqrt_LinearSystem(self):
        raise NotImplementedError
    
    def inv_LinearSystem(self):
        return InverseLinearSystem(self)
    
    def logdet(self):
        raise NotImplementedError
    
    def det(self):
        return torch.exp(self.logdet())
    
    def _pseudoinverse_weighted_average(self, y: torch.Tensor):
        """
        This method implements the pseudoinverse of the linear operator using a weighted average.

        parameters:
            y: torch.Tensor 
                The input tensor to the pseudoinverse_weighted_average of the linear operator.
        returns:
            x: torch.Tensor 
                The result of applying the pseudoinverse_weighted_average of the linear operator to the input tensor.
        """
        
        numerator = self.conjugate_transpose(y)
        
        denominator = self.conjugate_transpose(torch.ones_like(y))

        x = numerator / (denominator + 1e-10)  # Avoid division by zero
        
        return x

    def _pseudoinverse_conjugate_gradient(self, b, max_iter=1000, tol=1e-6, beta=1e-3, verbose=False):
        """
        This method implements the pseudoinverse of the linear operator using the conjugate gradient method.

        It solves the linear system (A^T A + beta * I) x = A^T b for x, where A is the linear operator.

        parameters:
            b: torch.Tensor of shape
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            max_iter: int
                The maximum number of iterations to run the conjugate gradient method.
            tol: float
                The tolerance for the conjugate gradient method.
            beta: float
                The regularization strength for the conjugate gradient method.
        returns:
            x_est: torch.Tensor
                The result of applying the pseudoinverse_conjugate_gradient of the linear operator to the input tensor.
        """
        ATb = self.conjugate_transpose(b)
        x_est = self._pseudo_inverse_weighted_average(b)
        
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
        This method implements the pseudo inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            method: str
                The method to use for computing the pseudo inverse. If None, the method is chosen automatically.
            kwargs: dict
                Keyword arguments to be passed to the method.
        """

        if method is None:
            method = 'conjugate_gradient'

        assert method in ['weighted_average', 'conjugate_gradient'], "The method should be either 'weighted_average' or 'conjugate_gradient'."

        if method == 'weighted_average':
            result =  self._pseudoinverse_weighted_average(y, **kwargs)
        elif method == 'conjugate_gradient':
            result =  self._pseudoinverse_conjugate_gradient(y, **kwargs)

        return result
    
    def mat_add(self, M):
        raise NotImplementedError
    def mat_sub(self, M):
        raise NotImplementedError
    def mat_mul(self, M):
        return self.forward(M)
    def __mul__(self, x):
        return NotImplementedError
    def __add__(self, M):
        return self.mat_add(M)
    def __sub__(self, M):
        return self.mat_sub(M)
    def __matmul__(self, M):
        return self.mat_mul(M)

    
class RealLinearSystem(LinearSystem):
    def __init__(self):
        """
        This is an abstract class for real linear operators.

        It inherits from LinearSystem.

        parameters:
            None
        """

        super(RealLinearSystem, self).__init__()

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the conjugate of the linear operator.

        parameters:
            x: torch.Tensor
                The input tensor to the conjugate of the linear operator.

        returns:
            conjugate: LinearSystem object
                The conjugate of the linear operator.

        """
        # for real linear operators, the conjugate is the same as the forward
        self.forward(x)        
    
    def conjugate_LinearSystem(self):
        return self.forward_LinearSystem()
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """ 
        This method returns the conjugate transpose of the linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to the conjugate_transpose of the linear operator.

        returns:
            z: torch.Tensor
                The conjugate_transpose of the linear operator.

        """
        # for real linear operators, the conjugate transpose is the same as the transpose
        return self.transpose(y)
    
    def conjugate_transpose_LinearSystem(self):
        return self.transpose_LinearSystem()
    

class SquareLinearSystem(LinearSystem):
    def __init__(self):
        """
        This is an abstract class for square linear operators.

        It inherits from LinearSystem.

        For square linear operators, the input and output shapes are the same.

        parameters:
            None
        """

        super(SquareLinearSystem, self).__init__()

    def compute_input_shape_given_output_shape(self, output_shape):
        """
        This method computes the input shape given the output shape.

        parameters:
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
        returns:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """
        # for square linear operators, the input and output shapes are the same
        return output_shape
    

class InvertibleLinearSystem(SquareLinearSystem):
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor 
                The input tensor to which the inverse linear operator should be applied.
        returns:
            x: torch.Tensor
                The result of applying the inverse linear operator to the input tensor.
        """
        if not hasattr(self, 'inverse_LinearSystem'):
            return self.inverse_LinearSystem().forward(y)
        else:
            raise NotImplementedError("For InvertibleLinearSystem, either the inverse method or the inverse_LinearSystem method must be implemented.")
    
    def inverse_LinearSystem(self):
        """
        This method returns the inverse of the linear operator.

        returns:
            inverse: LinearSystem object
                The inverse of the linear operator.
        """
        if not hasattr(self, 'inverse'):
            return InverseLinearSystem(self)
        else:
            raise NotImplementedError("For InvrtibleLinearSystem, either the inverse method or the inverse_LinearSystem method must be implemented.")
        
class UnitaryLinearSystem(InvertibleLinearSystem):
    def __init__(self):
        """
        This is an abstract class for unitary linear operators.

        It inherits from InvertibleLinearSystem.

        parameters:
            None
        """

        super(UnitaryLinearSystem, self).__init__()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor 
                The input tensor to which the inverse linear operator should be applied.
        returns:
            x: torch.Tensor
                The result of applying the inverse linear operator to the input tensor.
        """
        # for unitary linear operators, the inverse is the conjugate transpose
        return self.conjugate_transpose(y)

    def inverse_LinearSystem(self):
        return self.conjugate_transpose_LinearSystem()


class HermitianLinearSystem(SquareLinearSystem):
    def __init__(self):
        """
        This is an abstract class for Hermitian, or self-adjoint linear operators.

        It inherits from SquareLinearSystem.

        parameters:
            None
        """

        super(HermitianLinearSystem, self).__init__()

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.forward(y)
    
    def conjugate_transpose_LinearSystem(self):
        return self
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.transpose(x)
    
    def conjugate_LinearSystem(self):
        return self.transpose_LinearSystem()
    

class SymmetricLinearSystem(SquareLinearSystem):
    def __init__(self):
        """
        This is an abstract class for Symmetric linear operators.

        It inherits from SquareLinearSystem.

        parameters:
            None
        """

        super(SymmetricLinearSystem, self).__init__()

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear operator.
        returns:
            z: torch.Tensor
                The result of applying the transpose of the linear operator to the input tensor.
        """
        return self.forward(y)
    
    def cojugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            y: torch.Tensor 
                The input tensor to the adjoint of the linear operator.
        returns:
            z: torch.Tensor 
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return self.conjugate(y)
    
    def transpose_LinearSystem(self):
        return self
    
class Scalar(SymmetricLinearSystem, InvertibleLinearSystem):
    def __init__(self, scalar):
        """
        This class implements a scalar linear operator.

        It inherits from SymmetricLinearSystem.

        parameters:
            scalar: float
                The scalar to multiply the input tensor with.
        """

        super(Scalar, self).__init__()

        # if scalar is a float, convert it to a tensor
        if isinstance(scalar, (int, float)):
            scalar = torch.tensor(scalar)

        self.scalar = scalar

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor 
                The result of applying the linear operator to the input tensor.
        """

        for i, shape in enumerate(self.scalar.shape):
            assert x.shape[i] == shape or self.scalar.shape[i]==1, "The input tensor shape does not match the scalar shape."

        target_shape = [self.scalar.shape[i] for i in range(len(self.scalar.shape))]
        for i in range(len(x.shape) - len(self.scalar.shape)):
            target_shape = target_shape + [1,]

        return self.scalar.reshape(target_shape).to(x.device) * x
    
    def conjugate(self, x):
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return torch.conj(self.scalar) * x
    
    def inverse(self, y):
        if self.scalar == 0:
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return y / self.scalar
    
    def inverse_LinearSystem(self):
        if torch.any(self.scalar == 0):
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return Scalar(1/self.scalar)
    
    def sqrt_LinearSystem(self):
        return Scalar(torch.sqrt(self.scalar))

    def mat_add(self, added_scalar_matrix):
        assert isinstance(added_scalar_matrix, (Scalar)), "Scalar addition only supported for Scalar." 
        return Scalar(self.scalar + added_scalar_matrix.scalar)
    
    def mat_sub(self, sub_scalar_matrix):
        assert isinstance(sub_scalar_matrix, (Scalar)), "Scalar subtraction only supported for Scalar." 
        return Scalar(self.scalar - sub_scalar_matrix.scalar)

    def mat_mul(self, mul_scalar_matrix):
        if isinstance(mul_scalar_matrix, torch.Tensor):
            return self.forward(mul_scalar_matrix)
        elif isinstance(mul_scalar_matrix, Scalar):
            return Scalar(self.scalar * mul_scalar_matrix.scalar)
        else:
            raise ValueError("Unsupported type for matrix multiplication.")
        
    def logdet(self):
        return torch.log(torch.abs(self.scalar))
    


class DiagonalScalar(SymmetricLinearSystem, InvertibleLinearSystem):
    def __init__(self, diagonal_vector):
        """
        This class implements a diagonal linear operator.

        It inherits from SquareLinearSystem.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            diagonal_vector: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The diagonal of the linear operator.
        """

        super(DiagonalScalar, self).__init__()

        self.diagonal_vector = diagonal_vector
    
    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """
        return self.diagonal_vector * x
    
    def conjugate(self, x):
        return torch.conj(self.diagonal_vector) * x
    
    def inverse(self, y):
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return y / self.diagonal_vector
    
    def inverse_LinearSystem(self):
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return DiagonalScalar(self.input_shape, 1/self.diagonal_vector)
    
    def sqrt_LinearSystem(self):
        return DiagonalScalar(torch.sqrt(self.diagonal_vector))

    def mat_add(self, added_diagonal_matrix):
        assert isinstance(added_diagonal_matrix, (DiagonalScalar)), "DiagonalScalar addition only supported for DiagonalScalar." 
        assert self.input_shape == added_diagonal_matrix.input_shape, "DiagonalScalar addition only supported for DiagonalScalar with same input shape."
        return DiagonalScalar(self.diagonal_vector + added_diagonal_matrix.diagonal_vector)

    def mat_sub(self, sub_diagonal_matrix):
        assert isinstance(sub_diagonal_matrix, (DiagonalScalar)), "DiagonalScalar subtraction only supported for DiagonalScalar." 
        assert self.input_shape == sub_diagonal_matrix.input_shape, "DiagonalScalar subtraction only supported for DiagonalScalar with same input shape."
        return DiagonalScalar(self.diagonal_vector - sub_diagonal_matrix.diagonal_vector)
    
    def mat_mul(self, mul_diagonal_matrix):
        assert isinstance(mul_diagonal_matrix, (DiagonalScalar)), "DiagonalScalar multiplication only supported for DiagonalScalar." 
        assert self.input_shape == mul_diagonal_matrix.input_shape, "DiagonalScalar multiplication only supported for DiagonalScalar with same input shape."
        return DiagonalScalar(self.diagonal_vector * mul_diagonal_matrix.diagonal_vector)

class Identity(RealLinearSystem, UnitaryLinearSystem, HermitianLinearSystem, SymmetricLinearSystem):
    def __init__(self):
        """
        This class implements the identity linear operator.

        It inherits from SquareLinearSystem.

        parameters:
            None
        """

        SquareLinearSystem.__init__(self)

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor 
                The result of applying the linear operator to the input tensor.
        """
        return x
    



class ConjugateLinearSystem(LinearSystem):
    def __init__(self, base_matrix_operator: LinearSystem):
        """
        This is an abstract class for linear operators that are the conjugate of another linear operator.

        It inherits from LinearSystem.

        parameters:
            base_matrix_operator: LinearSystem object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, LinearSystem), "The linear operator should be a LinearSystem object."
        super(ConjugateLinearSystem, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(y)
    
class TransposeLinearSystem(LinearSystem):
    def __init__(self, base_matrix_operator: LinearSystem):
        """
        This is an abstract class for linear operators that are the transpose of another linear operator.

        It inherits from LinearSystem.

        parameters:
            base_matrix_operator: LinearSystem object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, LinearSystem), "The linear operator should be a LinearSystem object."

        super(TransposeLinearSystem, self).__init__()

        self.base_matrix_operator = base_matrix_operator  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)

class ConjugateTransposeLinearSystem(LinearSystem):
    def __init__(self, base_matrix_operator: LinearSystem):
        """
        This is an abstract class for linear operators that are the conjugate transpose of another linear operator.

        It inherits from LinearSystem.

        parameters:
            base_matrix_operator: LinearSystem object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, LinearSystem), "The linear operator should be a LinearSystem object."

        super(ConjugateTransposeLinearSystem, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)
    

class InverseLinearSystem(InvertibleLinearSystem):
    def __init__(self, base_matrix_operator: InvertibleLinearSystem):
        """
        This is an abstract class for linear operators that are the inverse of another linear operator.

        It inherits from SquareLinearSystem.

        parameters:
            base_matrix_operator: LinearSystem object
                The linear operator to which the inverse should be applied.
        """
        assert isinstance(base_matrix_operator, InvertibleLinearSystem), "The input linear operator should be a InvertibleLinearSystem object."

        super(InverseLinearSystem, self).__init__()

        self.base_matrix_operator = base_matrix_operator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.inverse(x)
    
    def inverse_LinearSystem(self):
        return self.base_matrix_operator.forward_LinearSystem()

class CompositeLinearSystem(LinearSystem):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from LinearSystem.

        parameters:
            operators: list of LinearSystem objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """

        assert isinstance(matrix_operators, list), "The operators should be provided as a list of LinearSystem objects."
        assert len(matrix_operators) > 0, "At least one operator should be provided."
        for operator in matrix_operators:
            assert isinstance(operator, LinearSystem), "All operators should be LinearSystem objects."

        LinearSystem.__init__(self)

        self.matrix_operators = matrix_operators

    def forward(self, x):
        result = x
        for matrix_operator in self.matrix_operators:
            result = matrix_operator.forward(result)
        return result
    
    def transpose(self,y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.transpose(result)
        return result
    
    def inverse(self, y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.inverse(result)
        return result
    




class InvertibleCompositeLinearSystem(CompositeLinearSystem, InvertibleLinearSystem):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from LinearSystem.

        parameters:
            operators: list of LinearSystem objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """
        for operator in matrix_operators:
            assert isinstance(operator, InvertibleLinearSystem), "All operators should be InvertibleLinearSystem objects."
        CompositeLinearSystem.__init__(self, matrix_operators)

    def inverse(self, y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.inverse(result)
        return result
    
    def inverse_LinearSystem(self):
        return CompositeLinearSystem([operator.inverse_LinearSystem() for operator in reversed(self.matrix_operators)])

class EigenDecomposedLinearSystem(CompositeLinearSystem, SymmetricLinearSystem):
    def __init__(self, eigenvalue_matrix: DiagonalScalar, eigenvector_matrix: InvertibleLinearSystem):
        """
        This class represents a linear operator that is given by its eigenvalue decomposition.

        It inherits from CompositeLinearSystem and SymmetricLinearSystem

        parameters:
            eigenvalue_matrix: DiagonalScalar object
                The diagonal matrix of eigenvalues.
            eigenvector_matrix: InvertibleLinearSystem object
                The invertible matrix of eigenvectors.
        """

        assert isinstance(eigenvalue_matrix, DiagonalScalar), "The eigenvalues should be a DiagonalScalar object."
        assert isinstance(eigenvector_matrix, InvertibleLinearSystem), "The eigenvectors should be a InvertibleLinearSystem object."

        self.eigenvalue_matrix = eigenvalue_matrix
        self.eigenvectors = eigenvector_matrix

        operators = [eigenvector_matrix, eigenvalue_matrix, eigenvector_matrix.inverse_LinearSystem()]

        CompositeLinearSystem.__init__(self, operators)







class SingularValueDecomposedLinearSystem(CompositeLinearSystem):
    def __init__(self, left_singular_vector_matrix: UnitaryLinearSystem, singular_value_matrix: DiagonalScalar,  right_singular_vector_matrix: UnitaryLinearSystem):
        """
        This class represents a linear operator that is given by its singular value decomposition.

        It inherits from SquareLinearSystem.

        parameters:
            singular_values: DiagonalScalar object
                The diagonal matrix of singular values.
            left_singular_vectors: UnitaryLinearSystem object
                The matrix of left singular vectors.
            right_singular_vectors: UnitaryLinearSystem object
                The matrix of right singular vectors.
        """

        assert isinstance(singular_value_matrix, DiagonalScalar), "The singular values should be a DiagonalScalar object."
        assert isinstance(left_singular_vector_matrix, UnitaryLinearSystem), "The left singular vectors should be a UnitaryLinearSystem object."
        assert isinstance(right_singular_vector_matrix, UnitaryLinearSystem), "The right singular vectors should be a UnitaryLinearSystem object."

        operators = [left_singular_vector_matrix, singular_value_matrix, right_singular_vector_matrix.conjugate_transpose_LinearSystem()]

        super(SingularValueDecomposedLinearSystem, self).__init__(operators)

        self.singular_values = singular_value_matrix
        self.left_singular_vectors = left_singular_vector_matrix
        self.right_singular_vectors = right_singular_vector_matrix

