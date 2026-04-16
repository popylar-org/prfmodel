"""Exceptions and warnings.

This module contains exceptions and warnings that are mostly for internal use.

"""

from collections.abc import Sequence


class BatchDimensionError(Exception):
    """
    Exception raised when arguments have different sizes in the batch (first) dimension.

    Parameters
    ----------
    arg_names: Sequence[str]
        Names of arguments that have different sizes in batch dimension.
    arg_shapes: Sequence[tuple of int]
        Shapes of arguments that have different sizes in batch dimension.

    """

    def __init__(self, arg_names: Sequence[str], arg_shapes: Sequence[tuple[int, ...]]):
        names = ", ".join(arg_names)
        shapes = ", ".join([str(s[0]) for s in arg_shapes])

        super().__init__(f"Arguments {names} have different sizes in batch (first) dimension: {shapes}")


class ShapeError(Exception):
    """
    Exception raised when an argument has less than two dimensions.

    Parameters
    ----------
    arg_name: str
        Argument name.
    arg_shape: tuple of int
        Argument shape.

    """

    def __init__(self, arg_name: str, arg_shape: tuple[int, ...]):
        super().__init__(
            f"Argument {arg_name} must have at least two dimensions but has shape {arg_shape}",
        )
