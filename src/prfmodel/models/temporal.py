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


class DivNormAmplitude(BaseTemporal):
    r"""
    Divisive normalization amplitude model.

    Combines two temporal responses using the divisive normalization formula. The ``- b/d`` term
    ensures a zero predicted response in the absence of a stimulus, which is appropriate for fMRI.
    For non-fMRI data you can set ``subtract_baseline=False`` to remove this correction.

    Parameters
    ----------
    subtract_baseline : bool, default=True
        If ``True`` (default), subtracts ``b/d`` from the output.

    Notes
    -----
    Given activation response :math:`r_1(t)` (``inputs[:, 0]``) and normalization response
    :math:`r_2(t)` (``inputs[:, 1]``), with :math:`a = \text{amplitude\_activation}`,
    :math:`b = \text{activation\_constant}`, :math:`c = \text{amplitude\_normalization}`, and
    :math:`d = \text{normalization\_constant}`, the predicted response is:

    .. math::

        p_{\text{DN}}(t) = \frac{a \, r_1(t) + b}{c \, r_2(t) + d} - \frac{b}{d}

    """

    def __init__(self, subtract_baseline: bool = True):
        self._subtract_baseline = subtract_baseline

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: ``amplitude_activation``, ``activation_constant`` (b in the DN formula),
            ``amplitude_normalization``, and ``normalization_constant`` (d in the DN formula).

        """
        return ["amplitude_activation", "activation_constant", "amplitude_normalization", "normalization_constant"]

    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with two temporal responses stacked along axis 1,
            shape (num_batches, 2, num_frames). Axis 1 index 0 is the activation response
            (G1·S) and index 1 is the normalization response (G2·S).
        parameters : pandas.DataFrame
            Dataframe with columns ``amplitude_activation``, ``activation_constant`` (b),
            ``amplitude_normalization``, and ``normalization_constant`` (d).
        dtype : str, optional
            The dtype of the prediction result. If ``None`` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions with shape (num_batches, num_frames) and dtype ``dtype``.

        """
        dtype = get_dtype(dtype)
        inputs = ops.convert_to_tensor(inputs, dtype=dtype)

        if len(inputs.shape) < _EXPECTED_NDIM:
            raise ShapeError(
                arg_name="inputs",
                arg_shape=inputs.shape,
            )

        amplitude_activation = convert_parameters_to_tensor(parameters[["amplitude_activation"]], dtype=dtype)
        b = convert_parameters_to_tensor(parameters[["activation_constant"]], dtype=dtype)
        amplitude_normalization = convert_parameters_to_tensor(parameters[["amplitude_normalization"]], dtype=dtype)
        d = convert_parameters_to_tensor(parameters[["normalization_constant"]], dtype=dtype)

        numerator = amplitude_activation * inputs[:, 0] + b
        denominator = amplitude_normalization * inputs[:, 1] + d
        result = numerator / denominator

        if self._subtract_baseline:
            result = result - b / d

        return result


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

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: ``amplitude_center``, ``amplitude_surround``, and ``baseline``.

        """
        return ["amplitude_center", "amplitude_surround", "baseline"]

    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with two temporal responses stacked along axis 1,
            shape (num_batches, 2, num_frames).
        parameters : pandas.DataFrame
            Dataframe with columns ``amplitude_center``, ``amplitude_surround``, and ``baseline``.
        dtype : str, optional
            The dtype of the prediction result. If ``None`` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions with shape (num_batches, num_frames) and dtype ``dtype``.

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
