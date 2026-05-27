"""Baseline scaling model."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.exceptions import ShapeError
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseScaling


class Baseline(BaseScaling):
    """
    Additive baseline scaling model.

    Transforms a temporal response by adding a baseline.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "baseline": [5.0, 10.0, -3.0],
    ... })
    >>> num_frames = 10
    >>> inputs = np.ones((params.shape[0], num_frames))
    >>> model = Baseline()
    >>> resp = model(inputs, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 10)

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `baseline`.

        """
        return ["baseline"]

    @doc
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : :data:`prfmodel.typing.Tensor`
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
            raise ShapeError("inputs", inputs.shape, f"must have exactly {_EXPECTED_NDIM} dimensions")  # noqa: EM101 (exception literal)

        baseline = convert_parameters_to_tensor(parameters[["baseline"]], dtype=dtype)

        return inputs + baseline
