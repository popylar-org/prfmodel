"""Adapters and parameter transformations."""

from collections.abc import Callable
from collections.abc import Sequence
import pandas as pd
from keras import ops
from prfmodel.fitters.backend.base import ParamsDict
from prfmodel.typing import Tensor


class ParameterTransform:
    """
    Apply transformations to parameters.

    Instances of this class can be used inside an :class:`Adapter` object to transform specific parameters during
    model fitting.

    Parameters
    ----------
    parameter_names : Sequence of str
        Names of parameters to be transformed.
    forward_fun : Callable
        Function to apply to parameters for the forward transformation. During model fitting, parameters will be
        optimized on the scale of the forward transformation (e.g., for a log-transformation, parameters will be
        optimized on the log-scale).
    inverse_fun : Callable
        Function to apply to parameters for the inverse transformation. Should be the inverse of `forward_fun`
        or the identity function (e.g., `lambda x: x`). During model fitting, model predictions will
        be made using parameters on the scale of the inverse-transformation (e.g., for a log-transformation, model
        predictions will be made with parameters on the natural scale).

    Notes
    -----
    When using the transform within stochastic gradient descent, the forward and inverse functions should allow for
    gradient tracking (e.g., by using functions from the `keras.ops` module).

    Examples
    --------
    Log-transform parameters.

    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "x": np.arange(1, 5)
    >>> })
    >>> transform = ParameterTransform(
    >>>     parameter_names=["x"],
    >>>     forward_fun=np.log,
    >>>     inverse_fun=np.exp,
    >>> )
    >>> params_forward = transform.forward(params)
    >>> print(params_forward)
              x
    0  0.000000
    1  0.693147
    2  1.098612
    3  1.386294

    Inverse transformation returns the original parameters for finite values.

    >>> params_inverse = transform.inverse(params_forward)
    >>> assert all(params_inverse == params)

    """

    def __init__(self, parameter_names: Sequence[str], forward_fun: Callable, inverse_fun: Callable):
        self.parameter_names = parameter_names
        self.forward_fun = forward_fun
        self.inverse_fun = inverse_fun

    def forward(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the forward transformation.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Dataframe with the forward transformation applied to the parameters specified in `parameter_names`.

        """
        parameters = parameters.copy()

        for param in self.parameter_names:
            parameters[param] = self.forward_fun(parameters[param])

        return parameters

    def inverse(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the inverse transformation.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

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
    """
    Constrain parameters to lower or upper bounds.

    Instances of this class can be used inside an :class:`Adapter` object to constrain specific parameters during
    model fitting using exponential transformation.

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
    transform_fun : Callable, optional
        Function to apply to the lower or upper bound before applying the constraint.

    Examples
    --------
    Constrain a parameter to be greater than another parameter.

    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "x": np.array([0.5, 1.0, 1.5]),
    >>>     "lower_bound": np.array([0.1, 0.2, 0.3])
    >>> })
    >>> constraint = ParameterContraint(
    >>>     parameter_names=["x"],
    >>>     lower="lower_bound",
    >>> )
    >>> params_forward = constraint.forward(params)
    >>> params_inverse = constraint.inverse(params_forward)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    Constrain a parameter to be greater than a fixed value.

    >>> constraint = ParameterContraint(
    >>>     parameter_names=["x"],
    >>>     lower=1.0,
    >>> )
    >>> params_forward = constraint.forward(params)
    >>> params_inverse = constraint.inverse(params_forward)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    Constrain a parameter to be greater than the square of another parameter.

    >>> constraint = ParameterContraint(
    >>>     parameter_names=["x"],
    >>>     lower="lower_bound",
    >>>     transform_fun=lambda x: x**2
    >>> )
    >>> params_forward = constraint.forward(params)
    >>> params_inverse = constraint.inverse(params_forward)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    """

    def __init__(
        self,
        parameter_names: Sequence[str],
        lower: str | float | None = None,
        upper: str | float | None = None,
        transform_fun: Callable | None = None,
    ):
        forward_fun = ops.exp
        inverse_fun = ops.log

        super().__init__(parameter_names, forward_fun, inverse_fun)

        if lower is not None and upper is not None:
            msg = "Lower and upper bound must not be provided at the same time"
            raise NotImplementedError(msg)

        self.lower = lower
        self.upper = upper

        if transform_fun is None:

            def identity(x: float | Tensor) -> float | Tensor:
                return x

            transform_fun = identity

        self.transform_fun = transform_fun

    def _forward_lower(self, parameters: pd.DataFrame) -> pd.DataFrame:
        parameters = parameters.copy()

        lower = parameters[self.lower] if isinstance(self.lower, str) else self.lower
        lower = self.transform_fun(lower)

        for param in self.parameter_names:
            parameters[param] = self.forward_fun(parameters[param]) + lower

        return parameters

    def _forward_upper(self, parameters: pd.DataFrame) -> pd.DataFrame:
        parameters = parameters.copy()

        upper = parameters[self.upper] if isinstance(self.upper, str) else self.upper
        upper = self.transform_fun(upper)

        for param in self.parameter_names:
            parameters[param] = -self.forward_fun(-parameters[param]) + upper

        return parameters

    def _inverse_lower(self, parameters: pd.DataFrame) -> pd.DataFrame:
        parameters = parameters.copy()

        lower = parameters[self.lower] if isinstance(self.lower, str) else self.lower
        lower = self.transform_fun(lower)

        for param in self.parameter_names:
            parameters[param] = self.inverse_fun(parameters[param] - lower)

        return parameters

    def _inverse_upper(self, parameters: pd.DataFrame) -> pd.DataFrame:
        parameters = parameters.copy()

        upper = parameters[self.upper] if isinstance(self.upper, str) else self.upper
        upper = self.transform_fun(upper)

        for param in self.parameter_names:
            parameters[param] = -self.inverse_fun(-parameters[param] + upper)

        return parameters

    def forward(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the forward constraint transformation.

        Transforms parameters by constraining them to be within specified bounds using exponential transformations.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Dataframe with the forward constraint transformation applied to the parameters specified in
            `parameter_names`.

        """
        is_dataframe = isinstance(parameters, pd.DataFrame)

        if is_dataframe:
            parameters = ParamsDict(parameters.to_dict(orient="list"))

        parameters = self._forward_lower(parameters) if self.lower is not None else self._forward_upper(parameters)

        if is_dataframe:
            return parameters.to_dataframe()

        return parameters

    def inverse(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the inverse constraint transformation.

        Transforms parameters back from the constrained space to the natural scale.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Dataframe with the inverse constraint transformation applied to the parameters specified in
            `parameter_names`.

        """
        is_dataframe = isinstance(parameters, pd.DataFrame)

        if is_dataframe:
            parameters = ParamsDict(parameters.to_dict(orient="list"))

        parameters = self._inverse_lower(parameters) if self.lower is not None else self._inverse_upper(parameters)

        if is_dataframe:
            return parameters.to_dataframe()

        return parameters


class Adapter:
    """Apply a series of transformations to parameters.

    Applies transformations sequentially to different parameters. This can be useful for model fitting to optimize
    parameters on a different scale instead of their natural one.

    Parameters
    ----------
    transforms : list of ParameterTransform, optional
        A list of :class:`ParameterTransform` or :class:`ParameterConstraint` objects that will be applied in the
        given order. If `None`, no transformations will be applied.

    Examples
    --------
    Apply multiple transformations to different parameters.

    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "x": np.arange(1, 5),
    >>>     "y": np.arange(2, 6)
    >>> })
    >>> transform_x = ParameterTransform(
    >>>     parameter_names=["x"],
    >>>     forward_fun=np.log,
    >>>     inverse_fun=np.exp,
    >>> )
    >>> transform_y = ParameterTransform(
    >>>     parameter_names=["y"],
    >>>     forward_fun=np.sqrt,
    >>>     inverse_fun=np.square,
    >>> )
    >>> adapter = Adapter(transforms=[transform_x, transform_y])
    >>> params_forward = adapter.forward(params)
    >>> params_inverse = adapter.inverse(params_forward)
    >>> assert all(params_inverse == params)

    """

    def __init__(self, transforms: list[ParameterTransform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = transforms

    def forward(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the forward transformations sequentially.

        Applies each forward transformation in the list of transforms to the parameters in order.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Transformed parameters after applying all forward transformations.

        """
        for transform in self.transforms:
            parameters = transform.forward(parameters)

        return parameters

    def inverse(self, parameters: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the inverse transformations sequentially.

        Applies each inverse transformation in the list of transforms to the parameters in reverse order.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Transformed parameters after applying all inverse transformations in reverse order.

        """
        for transform in reversed(self.transforms):
            parameters = transform.inverse(parameters)

        return parameters
