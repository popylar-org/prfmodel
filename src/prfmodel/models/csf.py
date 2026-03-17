"""Contrast sensitivity function response models."""

import math
import pandas as pd
from keras import ops
from prfmodel.stimuli.csf import CSFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseImpulse
from .base import BaseResponse
from .base import BaseTemporal
from .base import BatchDimensionError
from .base import ShapeError
from .composite import SimpleCSFModel
from .impulse import DerivativeTwoGammaImpulse
from .temporal import BaselineAmplitude


def predict_contrast_sensitivity(  # noqa: PLR0913 (too many arguments)
    sf: Tensor,
    cs_peak: Tensor,
    sf_peak: Tensor,
    width_l: float | Tensor,
    width_r: Tensor,
    dtype: str | None = None,
) -> Tensor:
    r"""
    Predict a neural contrast sensitivity function response.

    Computes the response using an asymmetric log-parabolic (aLP) contrast sensitivity function (CSF).

    Parameters
    ----------
    sf : Tensor
        Spatial frequency at each time frame, with shape ``(num_frames,)``.
    cs_peak : Tensor
        Peak contrast sensitivity with shape ``(num_batches, 1)``.
    sf_peak : Tensor
        Peak spatial frequency with shape ``(num_batches, 1)``.
    width_l : float or Tensor
        Curvature of the left branch of the aLP CSF (for spatial frequencies below ``sfp``).
        Scalar or tensor broadcastable to ``(num_batches, num_frames)``.
    width_r : Tensor
        Curvature of the right branch of the aLP CSF (for spatial frequencies at or above ``sfp``),
        with shape ``(num_batches, 1)``.
    dtype : str, optional
        The dtype of the prediction result. If ``None`` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.

    Returns
    -------
    Tensor
        Predicted response with shape ``(num_batches, num_frames)`` and dtype ``dtype``.

    Notes
    -----
    The asymmetric log-parabolic CSF :math:`S(f)` with `cs_peak` :math:`C_{sp}`, `sf_peak` :math:`f_p`,
    `width_l` :math:`w_L`, and `width_r` :math:`w_R` is:

    .. math::

        S(f) =
        \begin{cases}
        10^{\log_{10}(C_{sp}) - (\log_{10}(f) - \log_{10}(f_p))^2 \cdot w_L^2} & \text{if } f < f_p \\
        10^{\log_{10}(C_{sp}) - (\log_{10}(f) - \log_{10}(f_p))^2 \cdot w_R^2} & \text{if } f \geq f_p
        \end{cases}

    """
    dtype = get_dtype(dtype)

    for _name, _arg in [("cs_peak", cs_peak), ("sf_peak", sf_peak), ("width_r", width_r)]:
        if hasattr(_arg, "ndim") and _arg.ndim < _EXPECTED_NDIM:
            raise ShapeError(_name, tuple(_arg.shape))

    _batched = [
        (n, a)
        for n, a in [("csp", cs_peak), ("sfp", sf_peak), ("width_r", width_r)]
        if hasattr(a, "shape") and len(a.shape) >= _EXPECTED_NDIM and a.shape[0] != 1
    ]
    if len({a.shape[0] for _, a in _batched}) > 1:
        raise BatchDimensionError([n for n, _ in _batched], [tuple(a.shape) for _, a in _batched])

    log10 = math.log(10.0)

    sf = ops.expand_dims(ops.convert_to_tensor(sf, dtype=dtype), 0)  # (1, num_frames)

    cs_peak = ops.convert_to_tensor(cs_peak, dtype=dtype)  # (num_batches, 1)
    sf_peak = ops.convert_to_tensor(sf_peak, dtype=dtype)  # (num_batches, 1)
    width_l = ops.convert_to_tensor(width_l, dtype=dtype)  # scalar or (num_batches, 1)
    width_r = ops.convert_to_tensor(width_r, dtype=dtype)  # (num_batches, 1)

    # Log-domain: (num_batches, num_frames) via broadcasting
    log_sf = ops.log(sf) / log10
    log_sf_peak = ops.log(sf_peak) / log10
    log_cs_peak = ops.log(cs_peak) / log10
    log_diff_sq = ops.square(log_sf - log_sf_peak)

    # Asymmetric branches of the aLP CSF
    sensitivity_l = ops.power(10.0, log_cs_peak - log_diff_sq * ops.square(width_l))
    sensitivity_r = ops.power(10.0, log_cs_peak - log_diff_sq * ops.square(width_r))

    # Select branch: left where sf < sfp, right where sf >= sfp
    return ops.where(sf < sf_peak, sensitivity_l, sensitivity_r)


def predict_contrast_response(
    contrast: Tensor,
    sensitivity: Tensor,
    slope_crf: Tensor,
    dtype: str | None = None,
    eps: float = 1e-10,
) -> Tensor:
    r"""
    Predict a neural contrast response from contrast sensitivity.

    Computes the response using a contrast sensitivity function combined with a Naka-Rushton
    contrast response function (CRF).

    Parameters
    ----------
    contrast : Tensor
        Contrast at each time frame, with shape ``(num_frames,)``.
    sensitivity : Tensor
        Contrast sensitivity for each voxel at each time frame, with shape ``(num_voxels, num_frames)``.
    slope_crf : Tensor
        Exponent :math:`q` of the Naka-Rushton CRF with shape ``(num_batches, 1)``.
    dtype : str, optional
        The dtype of the prediction result. If ``None`` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.
    eps : float, default=1e-10
        Minimum contrast and sensitivity value to ensure numerical and gradient stability.

    Returns
    -------
    Tensor
        Predicted response with shape ``(num_batches, num_frames)`` and dtype ``dtype``.

    Notes
    -----
    The Naka-Rushton CRF combines contrast :math:`C` with the threshold :math:`Q = 1 / S(f)` derived from the CSF S(f):

    .. math::

        R(t) = \frac{C(t)^q}{C(t)^q + Q(t)^q}

    """
    dtype = get_dtype(dtype)

    for _name, _arg in [("sensitivity", sensitivity), ("slope_crf", slope_crf)]:
        if hasattr(_arg, "ndim") and _arg.ndim < _EXPECTED_NDIM:
            raise ShapeError(_name, tuple(_arg.shape))

    _batched = [
        (n, a)
        for n, a in [("sensitivity", sensitivity), ("slope_crf", slope_crf)]
        if hasattr(a, "shape") and len(a.shape) >= _EXPECTED_NDIM and a.shape[0] != 1
    ]
    if len({a.shape[0] for _, a in _batched}) > 1:
        raise BatchDimensionError([n for n, _ in _batched], [tuple(a.shape) for _, a in _batched])

    contrast = ops.expand_dims(ops.convert_to_tensor(contrast, dtype=dtype), 0)  # (1, num_frames)
    sensitivity = ops.convert_to_tensor(sensitivity, dtype=dtype)  # (num_batches, num_frames)
    slope_crf = ops.convert_to_tensor(slope_crf, dtype=dtype)  # (num_batches, 1)

    # Naka-Rushton: R = c^q / (c^q + Q^q)  where Q = 100 / sensitivity
    c_q = ops.power(ops.maximum(contrast, eps), slope_crf)
    q_q = ops.power(ops.maximum(100.0 / sensitivity, eps), slope_crf)

    return c_q / (c_q + q_q)


class CSFResponse(BaseResponse[CSFStimulus]):
    r"""
    Neural contrast sensitivity function response model.

    Predicts the response to a contrast sensitivity function stimulus using
    an asymmetric log-parabolic CSF combined with a Naka-Rushton contrast response function.

    Parameters
    ----------
    width_l : float, default=0.68
        Curvature of the left branch of the asymmetric log-parabolic CSF (for spatial frequencies
        below the peak). Not a fitted parameter; fixed at construction time.
    eps : float, default=1e-10
        Minimum contrast and sensitivity value to ensure numerical and gradient stability.

    Notes
    -----
    The model has four fitted parameters:

    ``cs_peak``
        Peak contrast sensitivity.
    ``sf_peak``
        Peak spatial frequency (cycles/degree) at which sensitivity is maximal.
    ``width_r``
        Curvature of the right branch of the aLP CSF.
    ``slope_crf``
        Exponent :math:`q` of the Naka-Rushton CRF, controlling response steepness.

    See Also
    --------
    predict_contrast_sensitivity : Predict a contrast sensitivity function.
    predict_contrast_response : Predict a contrast response based on a contrast sensitivity function.

    """

    def __init__(self, width_l: float = 0.68, eps: float = 1e-10):
        self.width_l = width_l
        self.eps = eps

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: ``cs_peak``, ``sf_peak``, ``width_r``, ``slope_crf``."""
        return ["cs_peak", "sf_peak", "width_r", "slope_crf"]

    def __call__(self, stimulus: CSFStimulus, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a CSF stimulus.

        Parameters
        ----------
        stimulus : CSFStimulus
            Contrast sensitivity function stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels. Must contain the columns ``cs_peak``, ``sf_peak``, ``width_r``, and ``slope_crf``.
        dtype : str, optional
            The dtype of the prediction result. If ``None`` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape ``(num_voxels, num_frames)`` and dtype ``dtype``.
            ``num_voxels`` is the number of rows in ``parameters`` and ``num_frames`` is the length of
            the stimulus sf and contrast arrays.

        """
        dtype = get_dtype(dtype)
        sf = ops.convert_to_tensor(stimulus.sf, dtype=dtype)
        contrast = ops.convert_to_tensor(stimulus.contrast, dtype=dtype)
        cs_peak = convert_parameters_to_tensor(parameters[["cs_peak"]], dtype=dtype)
        sf_peak = convert_parameters_to_tensor(parameters[["sf_peak"]], dtype=dtype)
        width_r = convert_parameters_to_tensor(parameters[["width_r"]], dtype=dtype)
        slope_crf = convert_parameters_to_tensor(parameters[["slope_crf"]], dtype=dtype)

        sensitivity = predict_contrast_sensitivity(
            sf=sf,
            cs_peak=cs_peak,
            sf_peak=sf_peak,
            width_l=self.width_l,
            width_r=width_r,
            dtype=dtype,
        )

        return predict_contrast_response(
            contrast=contrast,
            sensitivity=sensitivity,
            slope_crf=slope_crf,
            dtype=dtype,
            eps=self.eps,
        )


class CSFModel(SimpleCSFModel):
    """
    Contrast sensitivity function model.

    Convenience wrapper around :class:`~prfmodel.models.composite.SimpleCSFModel` with a
    :class:`CSFResponse` as the CSF model.

    Parameters
    ----------
    width_l : float, default=0.68
        Curvature of the left branch of the asymmetric log-parabolic CSF. Not fitted.
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse, optional
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=BaselineAmplitude, optional
        A temporal model class or instance.

    Notes
    -----
    The model follows four steps:

    1. The CSF response model predicts the response from per-frame spatial frequency and contrast.
    2. The impulse response model generates an impulse response.
    3. The CSF response is convolved with the impulse response.
    4. The temporal model modifies the convolved response.

    """

    def __init__(
        self,
        width_l: float = 0.68,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        super().__init__(
            csf_model=CSFResponse(width_l=width_l),
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )
