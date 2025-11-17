"""Adapters and parameter transformations."""

from collections.abc import Callable
from collections.abc import Sequence
import pandas as pd


class ParameterTransform:
    """
    Apply transformations to parameters.

    Instances of this class can be used inside an `Adapter` object to transform specific parameters during
    model fitting.

    Parameters
    ----------
    parameter_names : Sequence of str
        Names of parameters to be transformed.
    forward_fun : Callable
        Function to apply to parameters for the forward transformation. During model fitting, parameters will be
        optimized on the scale of the forward transformation (e.g., for a log-transformation, parameters will be
        optimzied on the log-scale).
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


class Adapter:
    """Apply a series of transformations to parameters.

    Applies transformations sequentially to different parameters. This can be useful for model fitting to optimize
    parameters on a different scale instead of their natural one.

    Parameters
    ----------
    transforms : list of ParameterTransform, optional
        A list of ParameterTransforms that will be applied in the given order. If `None`, no transformations will be
        applied.

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

        Applies each transformation in the list of transforms to the parameters in order.

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

        Applies each transformation in the list of transforms to the parameters in order.

        Parameters
        ----------
        parameters : pd.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.

        Returns
        -------
        pd.DataFrame
            Transformed parameters after applying all inverse transformations.

        """
        for transform in self.transforms:
            parameters = transform.inverse(parameters)

        return parameters
