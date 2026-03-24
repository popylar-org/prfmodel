"""Compressive spatial summation (CSS) population receptive field models."""

import numpy as np
import pandas as pd
from .base import BaseEncoder
from .base import BaseImpulse
from .base import BaseTemporal
from .encoding import CompressiveEncoder
from .encoding import PRFStimulusEncoder
from .gaussian import Gaussian2DPRFModel
from .impulse import DerivativeTwoGammaImpulse
from .temporal import BaselineAmplitude


class Gaussian2DCSSPRFModel(Gaussian2DPRFModel):
    """
    Two-dimensional isotropic Gaussian population receptive field model with compressive spatial summation (CSS).

    This is a generic class that combines a 2D isotropic Gaussian population receptive field, impulse,
    and temporal model response. In contrast to :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`, it
    encodes the stimulus response using compressive spatial summation (see
    :class:`~prfmodel.models.encoding.CompressiveEncoder`).

    Parameters
    ----------
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse, optional
        An impulse response model class or instance. Model classes will be instantiated during
        initialization. The default creates a :class:`~prfmodel.models.impulse.DerivativeTwoGammaImpulse`
        instance with default values.
    temporal_model : BaseTemporal or type or None, default=BaselineAmplitude, optional
        A temporal model class or instance. Model classes will be instantiated during initialization.
        The default creates a :class:`~prfmodel.models.temporal.BaselineAmplitude` instance.

    Notes
    -----
    The simple composite model follows five steps:

    1. The 2D Gaussian population receptive field response model makes a prediction for the stimulus grid.
    2. The encoding model encodes the response with the stimulus design and applies compressive spatial summation.
    3. A impulse response model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The temporal model modifies the convolved response.

    """

    def __init__(
        self,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        compressive_encoder: BaseEncoder = CompressiveEncoder(
            encoding_model=PRFStimulusEncoder(),
        )
        super().__init__(
            encoding_model=compressive_encoder,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )


def init_css_from_gaussian(gaussian_params: pd.DataFrame, gain: float = 1.0, n: float = 0.5) -> pd.DataFrame:
    """
    Initialize compressive spatial summation parameters from fitted Gaussian parameters.

    Converts the output of a fitted :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`
    into starting parameters for a :class:`Gaussian2DCSSPRFModel`.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a ``Gaussian2DPRFModel``.
    gain : float, default=1.0
        Amplification parameter for the :class:`~prfmodel.models.encoding.CompressiveEncoder`.
    n : float, default=0.5
        Compression exponent parameter for the :class:`~prfmodel.models.encoding.CompressiveEncoder`. Must be > 0.

    Returns
    -------
    pandas.DataFrame
        DataFrame with two additional columns: ``gain`` and ``m``.

    Raises
    ------
    ValueError
        If ``n`` <= 0 or if ``n`` or ``gain`` are not finite.

    """
    if not np.isfinite(n):
        msg = "'n' must be finite."
        raise ValueError(msg)
    if not np.isfinite(gain):
        msg = "'gain' must be finite."
        raise ValueError(msg)
    if n <= 0:
        msg = f"'n' must be > 0 but is {n}"
        raise ValueError(msg)

    css_params = gaussian_params.copy()
    css_params["gain"] = gain
    css_params["n"] = n

    return css_params
