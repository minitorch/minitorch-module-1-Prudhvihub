from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-06) -> Any:
    """
    Computes an approximation to the derivative of f with respect to one arg using
    central difference formula.
    
    Args:
        f: Function to differentiate
        *vals: Values to evaluate f at
        arg: Which argument to compute derivative with respect to
        epsilon: Small constant for approximation
        
    Returns:
        Approximation of f'_i(x_0, ..., x_{n-1})
    """
    # Convert vals to list so we can modify specific positions
    vals_list = list(vals)
    
    # Create x + epsilon
    vals_plus = vals_list.copy()
    vals_plus[arg] += epsilon
    
    # Create x - epsilon
    vals_minus = vals_list.copy()
    vals_minus[arg] -= epsilon
    
    # Apply central difference formula
    return (f(*vals_plus) - f(*vals_minus)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.
    
    Args:
        variable: The right-most variable
        
    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Keep track of visited nodes
    visited = set()
    sorted_variables = []
    
    def visit(var: Variable) -> None:
        # Skip if already visited or is constant
        if var.unique_id in visited or var.is_constant():
            return
            
        visited.add(var.unique_id)
        
        # Visit parents (dependencies) first
        if var.history is not None:
            for parent in var.parents:
                visit(parent)
                
        sorted_variables.append(var)
    
    visit(variable)
    return sorted_variables


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph to compute derivatives.
    
    Args:
        variable: The right-most variable
        deriv: Its derivative that we want to propagate backward to the leaves.
    """
    # Get variables in topological order
    sorted_variables = topological_sort(variable)
    
    # Dictionary to store derivatives for each variable
    derivatives = {variable.unique_id: deriv}
    
    # Iterate through variables in reverse topological order
    for var in reversed(sorted_variables):
        # Get the derivative for current variable
        deriv = derivatives[var.unique_id]
        
        # If it's a leaf node, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        # Otherwise, propagate to parents using chain rule
        elif var.history is not None:
            for parent_var, parent_deriv in var.chain_rule(deriv):
                parent_id = parent_var.unique_id
                if parent_id not in derivatives:
                    derivatives[parent_id] = parent_deriv
                else:
                    derivatives[parent_id] = derivatives[parent_id] + parent_deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
