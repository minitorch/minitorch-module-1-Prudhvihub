# Module 1: Autodifferentiation

This module implements a basic autodifferentiation system similar to PyTorch's autograd. Key components include:

- Scalar values with gradient tracking
- Basic mathematical operations (+, -, *, /, etc.)
- Common neural network functions (ReLU, Sigmoid)
- Backpropagation implementation
- Simple neural network training example

## Files
- `minitorch/`: Core autodiff implementation
  - `autodiff.py`: Base autodiff system
  - `scalar.py`: Scalar value implementation
  - `scalar_functions.py`: Mathematical operations
- `project/`: Training examples and visualization
  - `run_scalar.py`: Neural network training example

## Training Results
The implementation successfully trains on multiple datasets:
- Simple
- XOR
- Split
- Diag