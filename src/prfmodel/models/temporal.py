"""Temporal response models."""

import pandas as pd
from keras import ops
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
    >>>     "baseline": [5.0, 10.0, -3.0],
    >>>     "amplitude": [2.0, -1.0, 1.0],
    >>> })
    >>> num_frames = 10
    >>> inputs = np.ones((params.shape[0], num_frames))
    >>> model = BaselineAmplitude()
    >>> resp = model(inputs, params)
    >>> print(resp.shape) # (num_rows, num_frames)
    (3, 10)

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `baseline` and `amplitude`.

        """
        return ["baseline", "amplitude"]

    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with temporal response and shape (num_batches, num_frames).
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels. Must contain the columns `baseline` and `amplitude`.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions with the same shape as `inputs` and dtype `dtype`.

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
