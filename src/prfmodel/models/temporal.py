"""Temporal response models."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseTemporal
from .base import ShapeError


class BaselineAmplitude(BaseTemporal):
    """
    Linear baseline and amplitude model.

    Transforms a temporal response by multiplying it with an amplitude and adding a baseline.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "baseline": [5.0, 10.0, -3.0],
    ...     "amplitude": [2.0, -1.0, 1.0],
    ... })
    >>> num_frames = 10
    >>> inputs = np.ones((params.shape[0], num_frames))
    >>> model = BaselineAmplitude()
    >>> resp = model(inputs, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 10)

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `baseline` and `amplitude`.

        """
        return ["baseline", "amplitude"]

    @doc
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with temporal response and shape (num_units, num_frames).
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        inputs = ops.convert_to_tensor(inputs, dtype=dtype)

        if len(inputs.shape) != _EXPECTED_NDIM:
            raise ShapeError(
                arg_name="inputs",
                arg_shape=inputs.shape,
            )

        baseline = convert_parameters_to_tensor(parameters[["baseline"]], dtype=dtype)
        amplitude = convert_parameters_to_tensor(parameters[["amplitude"]], dtype=dtype)

        return inputs * amplitude + baseline


class DoGAmplitude(BaseTemporal):
    """
    Linear amplitude model for difference of Gaussians.

    Combines two temporal responses with independent amplitudes and a baseline:
    y(t) = inputs[:, 0] * amplitude_center + inputs[:, 1] * amplitude_surround + baseline

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: ``amplitude_center``, ``amplitude_surround``, and ``baseline``.

        """
        return ["amplitude_center", "amplitude_surround", "baseline"]

    @doc
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with two temporal responses stacked along axis 1,
            shape (num_units, 2, num_frames).
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        inputs = ops.convert_to_tensor(inputs, dtype=dtype)

        if len(inputs.shape) < _EXPECTED_NDIM:
            raise ShapeError(
                arg_name="inputs",
                arg_shape=inputs.shape,
            )

        amplitude_center = convert_parameters_to_tensor(parameters[["amplitude_center"]], dtype=dtype)
        amplitude_surround = convert_parameters_to_tensor(parameters[["amplitude_surround"]], dtype=dtype)
        baseline = convert_parameters_to_tensor(parameters[["baseline"]], dtype=dtype)

        return inputs[:, 0] * amplitude_center + inputs[:, 1] * amplitude_surround + baseline
