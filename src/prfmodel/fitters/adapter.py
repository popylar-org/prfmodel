"""Adapters and parameter transformations.

This module contains functionality to transform parameters during model fitting
(e.g., to optimize a parameter on the log scale). Currently, this is only implemented for SGD.

"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import TypeVar
from typing import cast
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame

P = TypeVar("P", pd.DataFrame, TensorFrame)


class ParameterTransform:
    """
    Apply transformations to parameters.

    Transforms parameter values using a custom function. Applies a custom function to invert the transformation.

    Parameters
    ----------
    parameter_names : Sequence of str
        Names of parameters to be transformed.
    transform_fun : Callable
        Function to apply to parameters for the transformation. During model fitting, parameters will be
        optimized on the scale of the transformation (e.g., for a log-transformation, parameters will be
        optimized on the log-scale).
    inverse_fun : Callable
        Function to apply to parameters for the inverse transformation. Should be the inverse of `transform_fun`
        or the identity function (e.g., `lambda x: x`). During model fitting, model predictions will
        be made using parameters on the scale of the inverse transformation (e.g., for a log-transformation, model
        predictions will be made with parameters on the natural scale).

    Notes
    -----
    Instances of this class can be used inside an :class:`~prfmodel.adapter.Adapter` object to transform specific
    parameters during model fitting.

    When using the transform within stochastic gradient descent, the transform and inverse functions should allow for
    gradient tracking (e.g., by using functions from the `keras.ops` module).

    Examples
    --------
    Log-transform parameters.

    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "x": np.arange(1, 5)
    ... })
    >>> transform = ParameterTransform(
    ...     parameter_names=["x"],
    ...     transform_fun=np.log,
    ...     inverse_fun=np.exp,
    ... )
    >>> params_transformed = transform.transform(params)
    >>> print(params_transformed)  # doctest: +NORMALIZE_WHITESPACE
              x
    0  0.000000
    1  0.693147
    2  1.098612
    3  1.386294

    Inverse transformation returns the original parameters for finite values.

    >>> params_inverse = transform.inverse(params_transformed)
    >>> assert all(params_inverse == params)

    """

    def __init__(self, parameter_names: Sequence[str], transform_fun: Callable, inverse_fun: Callable):
        self.parameter_names = parameter_names
        self.transform_fun = transform_fun
        self.inverse_fun = inverse_fun

    @doc
    def transform(self, parameters: P) -> P:
        """
        Apply the transformation.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        pd.DataFrame
            Dataframe with the transformation applied to the parameters specified in `parameter_names`.

        """
        parameters = parameters.copy()

        for param in self.parameter_names:
            parameters[param] = self.transform_fun(parameters[param])

        return parameters

    @doc
    def inverse(self, parameters: P) -> P:
        """
        Apply the inverse transformation.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        pd.DataFrame
            Dataframe with the inverse transformation applied to the parameters specified in `parameter_names`.

        """
        parameters = parameters.copy()

        for param in self.parameter_names:
            parameters[param] = self.inverse_fun(parameters[param])

        return parameters


class ParameterConstraint(ParameterTransform):
    r"""
    Constrain parameters to lower or upper bounds.

    Maps a bounded parameter onto an unbounded scale that a gradient-based optimizer can explore freely, and maps it
    back onto the bounded scale to make model predictions. :meth:`transform` takes a value on the natural (bounded)
    scale and returns its unbounded counterpart; :meth:`inverse` does the reverse. This is the same direction
    convention as :class:`ParameterTransform`, and it is what makes the bound hold: because :meth:`inverse` is
    applied before every model prediction, the model can only ever see values that satisfy the constraint, no matter
    where the optimizer moves.

    Parameters
    ----------
    parameter_names : Sequence of str
        Names of parameters to be transformed.
    lower : str or float, optional
        Lower bound of parameter constraint. If the argument has type `str`, it will use another parameter as the
        dynamic lower bound. An argument of type `float` will be used as a static lower bound.
    upper : str or float, optional
        Upper bound of parameter constraint. If the argument has type `str`, it will use another parameter as the
        dynamic upper bound. An argument of type `float` will be used as a static upper bound.
    bound_fun : Callable, optional
        Function to apply to the lower or upper bound before applying the constraint.

    Raises
    ------
    NotImplementedError
        When both a lower and upper bound are specified.

    Notes
    -----
    Instances of this class can be used inside an :class:`~prfmodel.fitters.adapter.Adapter` object to constrain
    specific parameters during model fitting using an exponential transformation. With a lower bound :math:`L`, the
    two directions are:

    .. math::

        \mathrm{transform}(x) = \log(x - L), \qquad \mathrm{inverse}(u) = e^{u} + L

    and with an upper bound :math:`U`:

    .. math::

        \mathrm{transform}(x) = -\log(U - x), \qquad \mathrm{inverse}(u) = -e^{-u} + U

    The bound is **open**: :meth:`transform` requires values that strictly satisfy the constraint and raises a
    :class:`ValueError` otherwise, because a value sitting exactly on the bound has no finite counterpart on the
    unbounded scale. Initial values must therefore be chosen strictly inside the bound, not on it.

    :meth:`inverse` is total: every finite input maps to a value satisfying the bound, which is what lets an
    optimizer explore freely without ever handing the model an invalid parameter. Note that in finite precision the
    exponential eventually underflows relative to the bound, so a value far enough out rounds onto the bound exactly
    rather than merely approaching it (beyond roughly 15 units for a bound of order 1 in `float32`). The bound is
    never crossed, but a parameter can become numerically equal to it, so a lower bound of exactly ``0.0`` on a
    parameter that is divided by, such as a Gaussian width, is better set to a small positive value.

    Examples
    --------
    Constrain a parameter to be greater than another parameter.

    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "x": np.array([0.5, 1.0, 1.5]),
    ...     "lower_bound": np.array([0.1, 0.2, 0.3])
    ... })
    >>> constraint = ParameterConstraint(
    ...     parameter_names=["x"],
    ...     lower="lower_bound",
    ... )
    >>> params_transformed = constraint.transform(params)
    >>> params_inverse = constraint.inverse(params_transformed)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    Constrain a parameter to be greater than a fixed value.

    >>> constraint = ParameterConstraint(
    ...     parameter_names=["x"],
    ...     lower=0.0,
    ... )
    >>> params_transformed = constraint.transform(params)
    >>> params_inverse = constraint.inverse(params_transformed)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    The transformed values are unbounded, and mapping any of them back always satisfies the constraint. This is what
    an optimizer relies on: it can propose any real number and still yield a valid parameter.

    >>> anywhere = pd.DataFrame({"x": np.array([-50.0, 0.0, 50.0])})
    >>> bool(np.all(constraint.inverse(anywhere)["x"] > 0.0))
    True

    Constrain a parameter to be greater than the square of another parameter.

    >>> constraint = ParameterConstraint(
    ...     parameter_names=["x"],
    ...     lower="lower_bound",
    ...     bound_fun=lambda x: x**2
    ... )
    >>> params_transformed = constraint.transform(params)
    >>> params_inverse = constraint.inverse(params_transformed)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    """

    def __init__(
        self,
        parameter_names: Sequence[str],
        lower: str | float | None = None,
        upper: str | float | None = None,
        bound_fun: Callable | None = None,
    ):
        # 'transform' maps the natural (bounded) scale onto the unbounded scale the optimizer works on,
        # 'inverse' maps back onto the natural scale that model predictions are made with
        transform_fun = ops.log
        inverse_fun = ops.exp

        super().__init__(parameter_names, transform_fun, inverse_fun)

        if lower is not None and upper is not None:
            msg = "Lower and upper bound must not be provided at the same time"
            raise NotImplementedError(msg)

        if lower is None and upper is None:
            msg = "Either a lower or an upper bound must be provided"
            raise ValueError(msg)

        self.lower = lower
        self.upper = upper

        if bound_fun is None:

            def identity(x: float | Tensor | None) -> float | Tensor | None:
                return x

            bound_fun = identity

        self.bound_fun = bound_fun

    def _check_bound_name(self, bound: str | float | None, parameters: TensorFrame) -> None:
        if isinstance(bound, str) and bound not in parameters.columns:
            msg = f"Parameters must contain the parameterized (dynamic) bound {bound}"
            raise ValueError(msg)

    def _check_strictly_inside(self, param: str, distance: Tensor) -> None:
        """Check that a parameter is strictly inside its bound before taking a logarithm.

        `distance` is the signed gap to the bound, which must be positive. A value on the bound has no finite
        counterpart on the unbounded scale, so this raises rather than silently producing an infinity that would
        surface much later as a NaN loss.

        """
        if not ops.all(ops.convert_to_tensor(distance) > 0.0):
            bound_name, bound = ("lower", self.lower) if self.lower is not None else ("upper", self.upper)
            msg = (
                f"Parameter '{param}' must be strictly {'greater' if bound_name == 'lower' else 'less'} than its "
                f"{bound_name} bound {bound!r}, but at least one value lies on or beyond it. The bound is open, so "
                f"choose starting values strictly inside it."
            )
            raise ValueError(msg)

    def _transform_lower(self, parameters: TensorFrame) -> TensorFrame:
        self._check_bound_name(self.lower, parameters)
        parameters = parameters.copy()

        bound = parameters[self.lower] if isinstance(self.lower, str) else self.lower
        # The bound cannot be None here: this method only runs when a lower bound was given, and
        # `__init__` rejects a constraint with no bound at all. `cast` tells the type checker that.
        lower = cast("Tensor | float", self.bound_fun(bound))

        for param in self.parameter_names:
            distance = parameters[param] - lower
            self._check_strictly_inside(param, distance)
            parameters[param] = self.transform_fun(distance)

        return parameters

    def _transform_upper(self, parameters: TensorFrame) -> TensorFrame:
        self._check_bound_name(self.upper, parameters)
        parameters = parameters.copy()

        bound = parameters[self.upper] if isinstance(self.upper, str) else self.upper
        # Cast for the same reason as in `_transform_lower`: an upper bound is guaranteed here.
        upper = cast("Tensor | float", self.bound_fun(bound))

        for param in self.parameter_names:
            distance = upper - parameters[param]
            self._check_strictly_inside(param, distance)
            parameters[param] = -self.transform_fun(distance)

        return parameters

    def _inverse_lower(self, parameters: TensorFrame) -> TensorFrame:
        self._check_bound_name(self.lower, parameters)
        parameters = parameters.copy()

        lower = parameters[self.lower] if isinstance(self.lower, str) else self.lower
        lower = self.bound_fun(lower)

        for param in self.parameter_names:
            parameters[param] = self.inverse_fun(parameters[param]) + lower

        return parameters

    def _inverse_upper(self, parameters: TensorFrame) -> TensorFrame:
        self._check_bound_name(self.upper, parameters)
        parameters = parameters.copy()

        upper = parameters[self.upper] if isinstance(self.upper, str) else self.upper
        upper = self.bound_fun(upper)

        for param in self.parameter_names:
            parameters[param] = -self.inverse_fun(-parameters[param]) + upper

        return parameters

    @doc
    def transform(self, parameters: P) -> P:
        """
        Map bounded parameters onto the unbounded scale.

        Takes parameters on their natural scale, where the bound holds, and returns their counterparts on the
        unbounded scale that a gradient-based optimizer works on. Use :meth:`inverse` for the reverse direction.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        pd.DataFrame
            Dataframe with the parameters in `parameter_names` mapped onto the unbounded scale.

        Raises
        ------
        ValueError
            If a dynamic bound is not a column in `parameters`, or if any value in `parameter_names` lies on or
            beyond its bound. The bound is open, so starting values must be strictly inside it.

        """
        if isinstance(parameters, pd.DataFrame):
            tensor_frame = TensorFrame(parameters.to_dict(orient="list"))
        else:
            tensor_frame = parameters

        tensor_frame = (
            self._transform_lower(tensor_frame) if self.lower is not None else self._transform_upper(tensor_frame)
        )

        if isinstance(parameters, pd.DataFrame):
            return tensor_frame.to_dataframe()

        return tensor_frame

    @doc
    def inverse(self, parameters: P) -> P:
        """
        Map unbounded parameters back onto the bounded natural scale.

        Takes parameters on the unbounded scale the optimizer works on and returns their counterparts on the natural
        scale, where the bound is guaranteed to hold for any finite input. This is the direction applied before every
        model prediction, which is what enforces the constraint during fitting.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        pd.DataFrame
            Dataframe with the parameters in `parameter_names` mapped onto the natural scale, satisfying the bound.

        Raises
        ------
        ValueError
            If a dynamic bound is not a column in `parameters`.

        """
        if isinstance(parameters, pd.DataFrame):
            tensor_frame = TensorFrame(parameters.to_dict(orient="list"))
        else:
            tensor_frame = parameters

        tensor_frame = (
            self._inverse_lower(tensor_frame) if self.lower is not None else self._inverse_upper(tensor_frame)
        )

        if isinstance(parameters, pd.DataFrame):
            return tensor_frame.to_dataframe()

        return tensor_frame


class Adapter:
    """Apply a series of transformations to parameters.

    Applies transformations sequentially to different parameters. This can be useful for model fitting to optimize
    parameters on a different scale instead of their natural one.

    Parameters
    ----------
    transforms : list of ParameterTransform, optional
        A list of :class:`~prfmodel.adapter.ParameterTransform` or :class:`~prfmodel.adapter.ParameterConstraint`
        objects that will be applied in the given order. If `None`, no transformations will be applied.

    Examples
    --------
    Apply multiple transformations to different parameters.

    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "x": np.arange(1, 5),
    ...     "y": np.arange(2, 6)
    ... })
    >>> transform_x = ParameterTransform(
    ...     parameter_names=["x"],
    ...     transform_fun=np.log,
    ...     inverse_fun=np.exp,
    ... )
    >>> transform_y = ParameterTransform(
    ...     parameter_names=["y"],
    ...     transform_fun=np.sqrt,
    ...     inverse_fun=np.square,
    ... )
    >>> adapter = Adapter(transforms=[transform_x, transform_y])
    >>> params_transformed = adapter.transform(params)
    >>> params_inverse = adapter.inverse(params_transformed)
    >>> assert all(params_inverse == params)

    """

    def __init__(self, transforms: list[ParameterTransform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = transforms

    @doc
    def transform(self, parameters: P) -> P:
        """
        Apply the transformations sequentially.

        Applies each transformation in the list of transforms to the parameters in order.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        pd.DataFrame
            Transformed parameters after applying all transformations.

        """
        for transform in self.transforms:
            parameters = transform.transform(parameters)

        return parameters

    @doc
    def inverse(self, parameters: P) -> P:
        """
        Apply the inverse transformations sequentially.

        Applies each inverse transformation in the list of transforms to the parameters in reverse order.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        pd.DataFrame
            Transformed parameters after applying all inverse transformations in reverse order.

        """
        for transform in reversed(self.transforms):
            parameters = transform.inverse(parameters)

        return parameters
