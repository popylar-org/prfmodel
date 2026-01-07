"""Adapters and parameter transformations."""

from collections.abc import Callable
from collections.abc import Sequence
from typing import TypeVar
import pandas as pd
from keras import ops
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict

P = TypeVar("P", pd.DataFrame, ParamsDict)


class ParameterTransform:
    """
    Apply transformations to parameters.

    Instances of this class can be used inside an :class:`Adapter` object to transform specific parameters during
    model fitting.

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
    When using the transform within stochastic gradient descent, the transform and inverse functions should allow for
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
    >>>     transform_fun=np.log,
    >>>     inverse_fun=np.exp,
    >>> )
    >>> params_transformed = transform.transform(params)
    >>> print(params_transformed)
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

    def transform(self, parameters: P) -> P:
        """
        Apply the transformation.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Dataframe with the transformation applied to the parameters specified in `parameter_names`.

        """
        parameters = parameters.copy()

        for param in self.parameter_names:
            parameters[param] = self.transform_fun(parameters[param])

        return parameters

    def inverse(self, parameters: P) -> P:
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
    bound_fun : Callable, optional
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
    >>> params_transformed = constraint.transform(params)
    >>> params_inverse = constraint.inverse(params_transformed)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    Constrain a parameter to be greater than a fixed value.

    >>> constraint = ParameterContraint(
    >>>     parameter_names=["x"],
    >>>     lower=1.0,
    >>> )
    >>> params_transformed = constraint.transform(params)
    >>> params_inverse = constraint.inverse(params_transformed)
    >>> assert np.allclose(params_inverse["x"], params["x"])

    Constrain a parameter to be greater than the square of another parameter.

    >>> constraint = ParameterContraint(
    >>>     parameter_names=["x"],
    >>>     lower="lower_bound",
    >>>     bound_fun=lambda x: x**2
    >>> )
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
        transform_fun = ops.exp
        inverse_fun = ops.log

        super().__init__(parameter_names, transform_fun, inverse_fun)

        if lower is not None and upper is not None:
            msg = "Lower and upper bound must not be provided at the same time"
            raise NotImplementedError(msg)

        self.lower = lower
        self.upper = upper

        if bound_fun is None:

            def identity(x: float | Tensor | None) -> float | Tensor | None:
                return x

            bound_fun = identity

        self.bound_fun = bound_fun

    def _check_bound_name(self, bound: str | float | None, parameters: ParamsDict) -> None:
        if isinstance(bound, str) and bound not in parameters.columns:
            msg = f"Parameters must contain the parameterized (dynamic) bound {bound}"
            raise ValueError(msg)

    def _transform_lower(self, parameters: ParamsDict) -> ParamsDict:
        self._check_bound_name(self.lower, parameters)
        parameters = parameters.copy()

        lower = parameters[self.lower] if isinstance(self.lower, str) else self.lower
        lower = self.bound_fun(lower)

        for param in self.parameter_names:
            parameters[param] = self.transform_fun(parameters[param]) + lower

        return parameters

    def _transform_upper(self, parameters: ParamsDict) -> ParamsDict:
        self._check_bound_name(self.upper, parameters)
        parameters = parameters.copy()

        upper = parameters[self.upper] if isinstance(self.upper, str) else self.upper
        upper = self.bound_fun(upper)

        for param in self.parameter_names:
            parameters[param] = -self.transform_fun(-parameters[param]) + upper

        return parameters

    def _inverse_lower(self, parameters: ParamsDict) -> ParamsDict:
        self._check_bound_name(self.lower, parameters)
        parameters = parameters.copy()

        lower = parameters[self.lower] if isinstance(self.lower, str) else self.lower
        lower = self.bound_fun(lower)

        for param in self.parameter_names:
            parameters[param] = self.inverse_fun(parameters[param] - lower)

        return parameters

    def _inverse_upper(self, parameters: ParamsDict) -> ParamsDict:
        self._check_bound_name(self.upper, parameters)
        parameters = parameters.copy()

        upper = parameters[self.upper] if isinstance(self.upper, str) else self.upper
        upper = self.bound_fun(upper)

        for param in self.parameter_names:
            parameters[param] = -self.inverse_fun(-parameters[param] + upper)

        return parameters

    def transform(self, parameters: P) -> P:
        """
        Apply the constraint transformation.

        Transforms parameters by constraining them to be within specified bounds using exponential transformations.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Dataframe with the constraint transformation applied to the parameters specified in
            `parameter_names`.

        """
        if isinstance(parameters, pd.DataFrame):
            param_dict = ParamsDict(parameters.to_dict(orient="list"))
        else:
            param_dict = parameters

        param_dict = self._transform_lower(param_dict) if self.lower is not None else self._transform_upper(param_dict)

        if isinstance(parameters, pd.DataFrame):
            return param_dict.to_dataframe()

        return param_dict

    def inverse(self, parameters: P) -> P:
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
        if isinstance(parameters, pd.DataFrame):
            param_dict = ParamsDict(parameters.to_dict(orient="list"))
        else:
            param_dict = parameters

        param_dict = self._inverse_lower(param_dict) if self.lower is not None else self._inverse_upper(param_dict)

        if isinstance(parameters, pd.DataFrame):
            return param_dict.to_dataframe()

        return param_dict


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
    >>>     transform_fun=np.log,
    >>>     inverse_fun=np.exp,
    >>> )
    >>> transform_y = ParameterTransform(
    >>>     parameter_names=["y"],
    >>>     transform_fun=np.sqrt,
    >>>     inverse_fun=np.square,
    >>> )
    >>> adapter = Adapter(transforms=[transform_x, transform_y])
    >>> params_transformed = adapter.transform(params)
    >>> params_inverse = adapter.inverse(params_transformed)
    >>> assert all(params_inverse == params)

    """

    def __init__(self, transforms: list[ParameterTransform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = transforms

    def transform(self, parameters: P) -> P:
        """
        Apply the transformations sequentially.

        Applies each transformation in the list of transforms to the parameters in order.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Transformed parameters after applying all transformations.

        """
        for transform in self.transforms:
            parameters = transform.transform(parameters)

        return parameters

    def inverse(self, parameters: P) -> P:
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
