import torch
from .composite import CompositeLinearSystem
from .invertible import InvertibleLinearSystem

class InvertibleCompositeLinearSystem(CompositeLinearSystem, InvertibleLinearSystem):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple invertible linear systems.
        
        It inherits from CompositeLinearSystem and InvertibleLinearSystem.
        
        parameters:
            matrix_operators: list of InvertibleLinearSystem objects
                The list of invertible linear systems to be composed. The product is taken in the order they are provided.
        """
        # Convert to list if it's a ListConfig from Hydra
        if hasattr(matrix_operators, '__iter__') and not isinstance(matrix_operators, list):
            matrix_operators = list(matrix_operators)
        
        assert isinstance(matrix_operators, list), "The operators should be provided as a list of InvertibleLinearSystem objects."
        assert len(matrix_operators) > 0, "At least one operator should be provided."
        
        # Instantiate operators if they're configs and validate they're invertible
        instantiated_operators = []
        for operator in matrix_operators:
            if hasattr(operator, '_target_'):
                # This is a config, need to instantiate it
                from hydra.utils import instantiate
                instantiated_operator = instantiate(operator)
                instantiated_operators.append(instantiated_operator)
            else:
                # This is already an operator
                instantiated_operators.append(operator)
            
            # Validate that all operators are invertible
            assert isinstance(instantiated_operators[-1], InvertibleLinearSystem), "All operators should be InvertibleLinearSystem objects."
        
        # Initialize the composite operator
        super().__init__(instantiated_operators)
    
    @property
    def is_invertible(self) -> bool:
        """
        Check if this composite linear system is invertible.
        
        A composite operator is invertible if and only if all its component operators are invertible.
        
        returns:
            bool: True if all component operators are invertible, False otherwise.
        """
        return all(operator.is_invertible for operator in self.matrix_operators)
    
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the invertible composite linear system.
        
        parameters:
            y: torch.Tensor
                The input tensor to the inverse of the invertible composite linear system.
        returns:
            result: torch.Tensor
                The result of applying the inverse of the invertible composite linear system to the input tensor.
        """
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.inverse(result)
        return result
    
    def inverse_LinearSystem(self):
        """
        This method returns the inverse linear system.
        
        returns:
            result: InvertibleCompositeLinearSystem
                The inverse linear system.
        """
        return InvertibleCompositeLinearSystem([operator.inverse_LinearSystem() for operator in reversed(self.matrix_operators)])


if __name__ == "__main__":
    def test_from_python():
        print("Testing InvertibleCompositeLinearSystem from Python...")
        try:
            from .scalar import Scalar
            from .diagonal import DiagonalScalar
            
            # Test with two invertible scalar operators
            op1 = Scalar(2.0)
            op2 = Scalar(3.0)
            composite = InvertibleCompositeLinearSystem([op1, op2])
            
            x = torch.tensor([1.0, 2.0, 3.0])
            
            # Test forward (should be 2.0 * 3.0 * x = 6.0 * x)
            y = composite.forward(x)
            expected = torch.tensor([6.0, 12.0, 18.0])
            assert torch.allclose(y, expected), f"Forward mismatch: {y} vs {expected}"
            print("SUCCESS: InvertibleCompositeLinearSystem forward works correctly")
            
            # Test inverse (should be (1/3.0) * (1/2.0) * x = (1/6.0) * x)
            x_recovered = composite.inverse(y)
            assert torch.allclose(x_recovered, x), f"Inverse mismatch: {x_recovered} vs {x}"
            print("SUCCESS: InvertibleCompositeLinearSystem inverse works correctly")
            
            # Test inverse_LinearSystem
            inverse_op = composite.inverse_LinearSystem()
            x_recovered2 = inverse_op.forward(y)
            assert torch.allclose(x_recovered2, x), f"Inverse operator mismatch: {x_recovered2} vs {x}"
            print("SUCCESS: InvertibleCompositeLinearSystem inverse_LinearSystem works correctly")
            
            # Test with diagonal and scalar operators
            diag_op = DiagonalScalar(torch.tensor([1.0, 2.0, 3.0]))
            scalar_op = Scalar(2.0)
            composite2 = InvertibleCompositeLinearSystem([diag_op, scalar_op])
            
            y2 = composite2.forward(x)
            expected2 = torch.tensor([2.0, 8.0, 18.0])  # diag([1,2,3]) * 2 * [1,2,3]
            assert torch.allclose(y2, expected2), f"Complex forward mismatch: {y2} vs {expected2}"
            print("SUCCESS: InvertibleCompositeLinearSystem with diagonal and scalar works correctly")
            
            # Test inverse of complex operator
            x_recovered3 = composite2.inverse(y2)
            assert torch.allclose(x_recovered3, x), f"Complex inverse mismatch: {x_recovered3} vs {x}"
            print("SUCCESS: InvertibleCompositeLinearSystem complex inverse works correctly")
            
            return True
        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_from_config():
        print("Testing InvertibleCompositeLinearSystem from config...")
        try:
            from hydra import compose, initialize
            from hydra.utils import instantiate
            with initialize(version_base=None, config_path="../../configs"):
                cfg = compose(config_name="linear_system/invertible_composite.yaml")
                op = instantiate(cfg.linear_system)
                
                # Test the instantiated operator
                x = torch.tensor([1.0, 2.0, 3.0])
                y = op.forward(x)
                expected = torch.tensor([6.0, 12.0, 18.0])  # 2.0 * 3.0 * x
                assert torch.allclose(y, expected), f"Config forward mismatch: {y} vs {expected}"
                
                # Test inverse
                x_recovered = op.inverse(y)
                assert torch.allclose(x_recovered, x), f"Config inverse mismatch: {x_recovered} vs {x}"
                print("SUCCESS: InvertibleCompositeLinearSystem from config works correctly")
                return True
        except Exception as e:
            print(f"ERROR: Unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    success_python = test_from_python()
    success_config = test_from_config()
    if success_python and success_config:
        print("All tests passed!")
    else:
        print("Some tests failed!") 