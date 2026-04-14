"""Divisive normalization amplitude scaling model."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.exceptions import ShapeError
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseTemporal


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

        Parameter names are: ``amplitude_activation``, ``baseline_activation`` (b in the DN formula),
            ``amplitude_normalization``, and ``baseline_normalization`` (d in the DN formula).

        """
        return ["amplitude_activation", "baseline_activation", "amplitude_normalization", "baseline_normalization"]

    @doc
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with two temporal responses stacked along axis 1,
            shape (num_batches, 2, num_frames). Axis 1 index 0 is the activation response
            (G1·S) and index 1 is the normalization response (G2·S).
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

        amplitude_activation = convert_parameters_to_tensor(parameters[["amplitude_activation"]], dtype=dtype)
        b = convert_parameters_to_tensor(parameters[["baseline_activation"]], dtype=dtype)
        amplitude_normalization = convert_parameters_to_tensor(parameters[["amplitude_normalization"]], dtype=dtype)
        d = convert_parameters_to_tensor(parameters[["baseline_normalization"]], dtype=dtype)

        numerator = amplitude_activation * inputs[:, 0] + b
        denominator = amplitude_normalization * inputs[:, 1] + d
        result = numerator / denominator

        if self._subtract_baseline:
            result = result - b / d

        return result
