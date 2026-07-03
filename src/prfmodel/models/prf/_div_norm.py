"""Divisive normalization population receptive field models."""

import warnings
import numpy as np
import numpy.typing as npt
import pandas as pd
from keras import ops
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.regressors.base import BaseRegressors
from prfmodel.scaling import Baseline
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import get_dtype
from ._gaussian import Gaussian2DPRFResponse
from ._stimulus_encoding import PRFStimulusEncoder
from .canonical import DivNormPRFModel


class DivNormGaussian2DPRFModel(DivNormPRFModel):
    r"""
    Divisive normalization population receptive field (pRF) model with isotropic 2D Gaussian responses.

    This class combines a divisive normalization 2D Gaussian response with an impulse,
    scaling, and regressors model. The two Gaussian 2D pRF models share the same center but have different sizes.

    Parameters
    ----------
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline, optional
        A scaling model class or instance. Model classes will be instantiated during initialization.
        The default creates a :class:`~prfmodel.scaling.Baseline` instance.
    %(model_regressors)s

    See Also
    --------
    init_div_norm_from_dog_css : Approximate good starting values for divisive normalization pRF models

    Notes
    -----
    The divisive normalization model follows these steps [1]_:

    1. The 2D Gaussian pRF response models make separate predictions for the stimulus grid.
    2. The encoding model encodes the responses with the stimulus design.
    3. The two encoded responses are combined through divisive normalization.
    4. The combined response is convolved with an impulse response (optional).
    5. The scaling model modifies the convolved response (optional).
    6. The regressors model adds a linear combination of fixed regressors to the scaled response (optional).

    Using the default impulse and scaling models, the following columns are expected in the
    :class:`pandas.DataFrame` passed as the ``parameters`` argument to :meth:`__call__`:

    .. list-table::
       :header-rows: 1
       :widths: 26 12 47

       * - Parameter
         - Model
         - Description
       * - ``mu_x``
         - pRF
         - Shared x-coordinate of both Gaussian centers.
       * - ``mu_y``
         - pRF
         - Shared y-coordinate of both Gaussian centers.
       * - ``sigma_activation``
         - pRF
         - Standard deviation of the activation Gaussian.
       * - ``sigma_normalization``
         - pRF
         - Standard deviation of the normalization Gaussian (must be >= ``sigma_activation``).
       * - ``delay``
         - Impulse
         - Peak time of the positive gamma component (in seconds; optional).
       * - ``dispersion``
         - Impulse
         - Rate parameter of the positive gamma component (optional).
       * - ``undershoot``
         - Impulse
         - Peak time of the negative gamma component (in seconds; optional).
       * - ``u_dispersion``
         - Impulse
         - Rate parameter of the negative gamma component (optional).
       * - ``ratio``
         - Impulse
         - Weight of the negative gamma component (optional).
       * - ``weight_deriv``
         - Impulse
         - Weight of the derivative component.
       * - ``amplitude_activation``
         - Scaling
         - Amplitude of the activation response (:math:`a`).
       * - ``baseline_activation``
         - Scaling
         - Baseline of the activation response (:math:`b`).
       * - ``amplitude_normalization``
         - Scaling
         - Amplitude of the normalization response (:math:`c`).
       * - ``baseline_normalization``
         - Scaling
         - Baseline of the normalization response (:math:`d`; must be > 0).

    References
    ----------
    .. [1] Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures
    throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118.
    https://doi.org/10.1073/pnas.2108713118

    """

    def __init__(
        self,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        super().__init__(
            prf_model=Gaussian2DPRFResponse(),
            shared_params=["mu_x", "mu_y"],
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )


def init_div_norm_from_dog_css(
    dog_params: pd.DataFrame,
    css_n: npt.ArrayLike = 0.5,
    stimulus: PRFStimulus | None = None,
) -> pd.DataFrame:
    r"""
    Initialize divisive normalization model parameters.

    Aims to provide good starting parameters for divisive normalization models by approximating
    suppression from Difference of Gaussian (DoG) model parameters and compression from
    compressive spatial summation (CSS) model parameters.

    Parameters
    ----------
    dog_params : pandas.DataFrame
        Parameter estimates from a DoG model (e.g., :class:`~prfmodel.models.prf.DoG2DPRFModel`). Columns contain
        different parameters and rows parameter values for different units.
    css_n : float or array_like, default=0.5
        Compression exponent from a CSS model (expected in the interval ``(0, 1)``).
        Smaller values give higher compression resulting from lower baseline normalization. Must be a scalar or of
        length equal to the rows in ``parameters``.
    stimulus : prfmodel.stimuli.PRFStimulus, optional
        Stimulus used to scale ``baseline_normalization`` to the actual normalization drive. When provided, the
        normalization Gaussian is encoded against the stimulus to obtain the peak normalization drive per row, and
        ``baseline_normalization`` is set so the local compression exponent of the divisive normalization response
        matches ``css_n`` (see Notes). When omitted, the cruder ``baseline_normalization = css_n * 100`` heuristic is
        used instead.

    Returns
    -------
    pandas.DataFrame
        Parameters required by divisive normalization models.

    See Also
    --------
    DivNormGaussian2DPRFModel : 2D Gaussian divisive normalization pRF model.
    prfmodel.models.prf.canonical.DivNormPRFModel : Generic divisive normalization pRF model.

    Notes
    -----
    The divisive normalization parameters are computed as follows:

    - ``sigma_activation = sigma_center``
    - ``sigma_normalization = sigma_surround``
    - ``amplitude_activation = 1.0``
    - ``amplitude_normalization = 1.0``
    - ``baseline_normalization = peak_normalization_drive * css_n / (1 - css_n)`` if ``stimulus`` is given, else
      ``css_n * 100`` (``peak_normalization_drive`` is the maximum of the stimulus-encoded normalization response)
    - ``baseline_activation = (amplitude_surround / amplitude_center) * baseline_normalization``

    Warns
    -----
    UserWarning
        If ``sigma_surround / sigma_center`` is less than 1.0 (normalization field smaller than activation field),
        if ``amplitude_center`` or ``amplitude_surround`` is negative (negative ``baseline_activation`` for positive
        ``amplitude``), if ``css_n`` is negative (negative ``baseline_normalization``), or if ``css_n`` is greater
        than or equal to 1.0 while ``stimulus`` is given (non-positive ``baseline_normalization``).

    """
    css_n = np.asarray(css_n)

    num_rows = dog_params.shape[0]
    if css_n.ndim > 0 and css_n.shape[0] not in (1, num_rows):
        msg = f"'css_n' must be a single value or of length equal to the number of rows in 'dog_params' ({num_rows})"
        raise ValueError(msg)

    dog_sigma_ratio = np.asarray(dog_params["sigma_surround"] / dog_params["sigma_center"])
    dog_amplitude_ratio = np.asarray(dog_params["amplitude_surround"] / dog_params["amplitude_center"])

    if np.any(dog_sigma_ratio < 1.0):
        warnings.warn(
            "'dog_sigma_ratio' is less than 1.0, giving a normalization field smaller than the activation field.",
            stacklevel=2,
        )
    if np.any(dog_amplitude_ratio < 0.0):
        warnings.warn(
            "'dog_amplitude_ratio' is negative, giving a negative baseline_activation for positive amplitude.",
            stacklevel=2,
        )
    if np.any(css_n < 0.0):
        warnings.warn(
            "'css_n' is negative, giving a negative baseline_normalization.",
            stacklevel=2,
        )
    if stimulus is not None and np.any(css_n >= 1.0):
        warnings.warn(
            "'css_n' is greater than or equal to 1.0, giving a non-positive baseline_normalization.",
            stacklevel=2,
        )

    div_norm_params = dog_params.drop(
        columns=["sigma_center", "sigma_surround", "amplitude_center", "amplitude_surround"],
    )

    div_norm_params["sigma_activation"] = dog_params["sigma_center"]
    div_norm_params["sigma_normalization"] = dog_params["sigma_surround"]

    div_norm_params["amplitude_activation"] = 1.0
    div_norm_params["amplitude_normalization"] = 1.0

    if stimulus is None:
        div_norm_params["baseline_normalization"] = np.broadcast_to(css_n * 100, div_norm_params.shape[0])
    else:
        peak_drive = _peak_normalization_drive(div_norm_params, stimulus)
        # css_n >= 1.0 (warned above) yields a non-positive denominator; ignore the numpy warning here.
        amplitude_normalization = div_norm_params["amplitude_normalization"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            div_norm_params["baseline_normalization"] = amplitude_normalization * peak_drive * css_n / (1.0 - css_n)

    div_norm_params["baseline_activation"] = dog_amplitude_ratio * div_norm_params["baseline_normalization"]

    return div_norm_params


def _peak_normalization_drive(div_norm_params: pd.DataFrame, stimulus: PRFStimulus) -> np.ndarray:
    """Peak normalization drive per row from encoding the normalization Gaussian against the stimulus."""
    norm_params = div_norm_params[["mu_x", "mu_y"]].copy()
    norm_params["sigma"] = div_norm_params["sigma_normalization"]

    dtype = get_dtype(None)
    norm_response = Gaussian2DPRFResponse()(stimulus, norm_params, dtype=dtype)
    norm_drive = PRFStimulusEncoder()(stimulus, norm_response, norm_params, dtype=dtype)

    return np.asarray(ops.convert_to_numpy(ops.max(norm_drive, axis=1)))
