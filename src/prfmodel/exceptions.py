"""Exceptions and warnings.

This module contains exceptions and warnings that are mostly for internal use.

"""


class ShapeError(ValueError):
    """
    Exception raised when an argument has an invalid shape.

    Parameters
    ----------
    arg_name : str
        Name of the argument.
    arg_shape : tuple of int
        Shape of the argument.
    requirement : str
        Description of the shape requirement that was violated (e.g. ``"must have at least 2 dimensions"``).

    """

    def __init__(self, arg_name: str, arg_shape: tuple[int, ...], requirement: str):
        super().__init__(f"Argument '{arg_name}' with shape {arg_shape} {requirement}")


class ShapeMismatchError(ValueError):
    """
    Exception raised when two arguments have incompatible shapes.

    Parameters
    ----------
    arg1_name : str
        Name of the first argument.
    arg1_shape : tuple of int
        Shape of the first argument.
    arg2_name : str
        Name of the second argument.
    arg2_shape : tuple of int
        Shape of the second argument.

    """

    def __init__(
        self,
        arg1_name: str,
        arg1_shape: tuple[int, ...],
        arg2_name: str,
        arg2_shape: tuple[int, ...],
    ):
        super().__init__(f"Shapes of '{arg1_name}' {arg1_shape} and '{arg2_name}' {arg2_shape} do not match")
