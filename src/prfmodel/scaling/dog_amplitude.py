"""Difference of Gaussian amplitude scaling model."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.exceptions import ShapeError
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseTemporal


class DoGAmplitude(BaseTemporal):
    r"""
    Linear amplitude model for difference of Gaussians.

    Combines two temporal responses with independent amplitudes and a baseline.

    Notes
    -----
    Given center response :math:`r_c(t)` (``inputs[:, 0]``) and surround response :math:`r_s(t)`
    (``inputs[:, 1]``), with :math:`a_c = \text{amplitude\_center}`,
    :math:`a_s = \text{amplitude\_surround}`, and :math:`\beta = \text{baseline}`, the predicted
    response is:

    .. math::

        y(t) = a_c \, r_c(t) + a_s \, r_s(t) + \beta

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "amplitude_center": [1.0, -1.0],
    ...     "amplitude_surround": [-0.5, 0.5],
    ...     "baseline": [0.0, 0.1],
    ... })
    >>> num_frames = 10
    >>> inputs = np.ones((params.shape[0], 2, num_frames))
    >>> model = DoGAmplitude()
    >>> resp = model(inputs, params)
    >>> print(resp.shape)
    (2, 10)

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
        inputs : :data:`prfmodel.typing.Tensor`
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
