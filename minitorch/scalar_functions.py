from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Log(ScalarFunction):
    "Log function $f(x) = log(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


class Mul(ScalarFunction):
    """
    Multiplication function f(x, y) = x * y
    """
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """
    Inverse function f(x) = 1/x
    """
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return 1.0 / a
    
    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float]:
        (a,) = ctx.saved_values
        return (-1.0 / (a * a) * d_output,)


class Neg(ScalarFunction):
    """
    Negation function f(x) = -x
    """
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -d_output


class Sigmoid(ScalarFunction):
    """
    Sigmoid function f(x) = 1 / (1 + e^(-x))
    """
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        # Compute sigmoid and save for backward pass
        sigmoid_val = 1.0 / (1.0 + operators.exp(-a))
        ctx.save_for_backward(sigmoid_val)
        return sigmoid_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        sigmoid_val, = ctx.saved_values
        return d_output * sigmoid_val * (1.0 - sigmoid_val)


class ReLU(ScalarFunction):
    """
    ReLU function f(x) = max(0, x)
    """
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return max(0.0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        a, = ctx.saved_values
        return d_output if a > 0.0 else 0.0


class Exp(ScalarFunction):
    """
    Exponential function f(x) = e^x
    """
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        exp_val = operators.exp(a)
        ctx.save_for_backward(exp_val)
        return exp_val

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        exp_val, = ctx.saved_values
        return exp_val * d_output


class LT(ScalarFunction):
    """
    Less than function f(x, y) = 1.0 if x < y else 0.0
    """
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Derivative is 0 since step functions have 0 gradient
        return 0.0, 0.0


class EQ(ScalarFunction):
    """
    Equal function f(x, y) = 1.0 if x == y else 0.0
    """
    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        # Derivative is 0 since step functions have 0 gradient
        return 0.0, 0.0
