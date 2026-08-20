"""Compressive spatiotemporal (CST) population receptive field models."""

import pandas as pd
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import SustainedImpulse
from prfmodel.impulse import TransientImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.regressors.base import BaseRegressors
from prfmodel.scaling import Baseline
from prfmodel.scaling.base import BaseScaling
from ._gaussian import Gaussian2DPRFResponse
from ._stimulus_encoding import PRFStimulusEncoder
from .canonical import CSTPRFModel


class Gaussian2DCSTPRFModel(CSTPRFModel):
    r"""
    Compressive spatiotemporal (CST) pRF model with a 2D isotropic Gaussian response.

    Thin wrapper around :class:`CSTPRFModel` that hardcodes
    :class:`~prfmodel.models.prf.Gaussian2DPRFResponse` as the pRF model.

    Parameters
    ----------
    %(model_encoding_prf)s
    sustained_model : BaseImpulse or type, default=SustainedImpulse
        Temporal channel model producing the sustained response h₁.
    transient_model : BaseImpulse or type, default=TransientImpulse
        Temporal channel model producing the on-transient response h₂. The off-transient h₃ is its negation.
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline
        Scaling model applied after the channels are combined.
    %(model_regressors)s

    Warnings
    --------
    **The temporal channels are sampled at the stimulus frame rate.** The reference computes the channels on a
    10 ms grid and downsamples to the repetition time; this package has no stimulus time axis, so one stimulus
    frame is one output sample throughout, and the channel models are given ``resolution=tr`` in the same way
    the impulse model is.

    The consequence is concrete: the reference reports ``time_to_peak`` values of roughly 50-110 ms in V1 rising
    to 160-230 ms in IPS, all of which fall inside a single frame at a one second repetition time. At that
    sampling the three channels are nearly indistinguishable and the temporal estimates the model exists to
    recover cannot be recovered. Use this model to build and check a pipeline, not to interpret temporal
    parameters, until fine-grained sampling is available.

    Notes
    -----
    Two deliberate differences from the reference:

    - **Sampling** — see the warning above.
    - **Spatial normalization** — the reference uses an unnormalized 2D Gaussian, while
      :func:`~prfmodel.models.prf.predict_gaussian_response` divides by the Gaussian volume. The difference is a
      constant per unit that is absorbed into the channel weights, so predictions are equivalent up to
      ``beta_sustained`` and ``beta_transient``, but fitted weights are not directly comparable to reported ones.

    The following columns are expected in the :class:`pandas.DataFrame` passed as the ``parameters`` argument to
    :meth:`__call__`:

    .. list-table::
       :header-rows: 1
       :widths: 22 12 12 49

       * - Parameter
         - Model
         - Paper symbol
         - Description
       * - ``mu_x``
         - pRF
         - --
         - x-coordinate of the Gaussian centre.
       * - ``mu_y``
         - pRF
         - --
         - y-coordinate of the Gaussian centre.
       * - ``sigma``
         - pRF
         - s
         - Standard deviation of the isotropic Gaussian, reported as the pRF size.
       * - ``time_to_peak``
         - h₁, h₂
         - τ
         - Peak time of the sustained channel (in seconds). Shared by both channels. Bounds in the reference are
           0.04 to 1.0 s, with fitted values of roughly 0.05 to 0.23 s.
       * - ``n``
         - --
         - n
         - Compressive exponent, shared by all three channels. Bounds in the reference are 0.1 to 1.
       * - ``beta_sustained``
         - --
         - β_sus
         - Weight of the sustained channel.
       * - ``beta_transient``
         - --
         - β_tran
         - Weight of the summed on- and off-transient channels.
       * - ``delay``
         - Impulse
         - --
         - Peak time of the positive gamma component (in seconds).
       * - ``dispersion``
         - Impulse
         - --
         - Rate parameter of the positive gamma component.
       * - ``undershoot``
         - Impulse
         - --
         - Peak time of the negative gamma component (in seconds).
       * - ``u_dispersion``
         - Impulse
         - --
         - Rate parameter of the negative gamma component.
       * - ``ratio``
         - Impulse
         - --
         - Weight of the negative gamma component.
       * - ``weight_deriv``
         - Impulse
         - --
         - Weight of the derivative component.
       * - ``baseline``
         - Scaling
         - --
         - Additive output constant.

    References
    ----------
    .. [1] Kim, I., Kupers, E. R., Lerma-Usabiaga, G., & Grill-Spector, K. (2024). Characterizing spatiotemporal
        population receptive fields in human visual cortex with fMRI. *The Journal of Neuroscience*, 44(2),
        e0803232023. https://doi.org/10.1523/JNEUROSCI.0803-23.2023

    Examples
    --------
    Predict a model response for multiple units.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> model = Gaussian2DCSTPRFModel()
    >>> params = pd.DataFrame({
    ...     "mu_x": [0.0, 1.0],
    ...     "mu_y": [1.0, 0.0],
    ...     "sigma": [1.0, 1.5],
    ...     "time_to_peak": [4.0, 5.0],
    ...     "n": [0.5, 0.8],
    ...     "beta_sustained": [1.0, 0.5],
    ...     "beta_transient": [0.5, 1.0],
    ...     "delay": [6.0, 6.0],
    ...     "dispersion": [0.9, 0.9],
    ...     "undershoot": [12.0, 12.0],
    ...     "u_dispersion": [0.9, 0.9],
    ...     "ratio": [0.48, 0.48],
    ...     "weight_deriv": [0.0, 0.0],
    ...     "baseline": [0.0, 0.0],
    ... })
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (2, 170)

    """

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        sustained_model: BaseImpulse | type[BaseImpulse] = SustainedImpulse,
        transient_model: BaseImpulse | type[BaseImpulse] = TransientImpulse,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        super().__init__(
            prf_model=Gaussian2DPRFResponse(),
            encoding_model=encoding_model,
            sustained_model=sustained_model,
            transient_model=transient_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )


def init_cst_from_gaussian(
    gaussian_params: pd.DataFrame,
    time_to_peak: float = 0.05,
    n: float = 0.5,
    beta_sustained: float = 1.0,
    beta_transient: float = 0.0,
) -> pd.DataFrame:
    """
    Initialize compressive spatiotemporal parameters from fitted Gaussian parameters.

    Converts the output of a fitted :class:`~prfmodel.models.prf.Gaussian2DPRFModel` into starting parameters for
    a :class:`Gaussian2DCSTPRFModel`, suitable for subsequent fitting. All existing columns (pRF, impulse,
    and scaling parameters) pass through unchanged. The four CST-specific parameters are appended with their
    default values.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a :class:`~prfmodel.models.prf.Gaussian2DPRFModel`.
    time_to_peak : float, default=0.05
        Peak time of the sustained channel (in seconds). The default is the grid search default of the
        reference (0.0493 s), which is also its reported V1 estimate.
    n : float, default=0.5
        Compressive exponent, in the middle of the reference bounds of 0.1 to 1.
    beta_sustained : float, default=1.0
        Weight of the sustained channel.
    beta_transient : float, default=0.0
        Weight of the transient channels. Defaults to zero so that the starting prediction is the sustained
        channel alone, which is the closest match to the Gaussian model the parameters come from.

    Returns
    -------
    pandas.DataFrame
        Copy of ``gaussian_params`` with four additional columns: ``time_to_peak``, ``n``, ``beta_sustained``,
        ``beta_transient``.

    Notes
    -----
    ``amplitude`` is not consumed by :class:`Gaussian2DCSTPRFModel`, whose default scaling model is
    :class:`~prfmodel.scaling.Baseline`; the channel weights take that role instead. Any ``amplitude`` column is
    passed through untouched and ignored.

    See the warning on :class:`Gaussian2DCSTPRFModel` before interpreting a fitted ``time_to_peak``.

    Examples
    --------
    >>> import pandas as pd
    >>> gaussian_params = pd.DataFrame({
    ...     "mu_x": [0.0, 1.0],
    ...     "mu_y": [0.0, -1.0],
    ...     "sigma": [1.0, 2.0],
    ...     "baseline": [0.0, 0.1],
    ... })
    >>> cst_params = init_cst_from_gaussian(gaussian_params)
    >>> print(sorted(cst_params.columns.tolist()))
    ['baseline', 'beta_sustained', 'beta_transient', 'mu_x', 'mu_y', 'n', 'sigma', 'time_to_peak']

    """
    cst_params = gaussian_params.copy()
    cst_params["time_to_peak"] = time_to_peak
    cst_params["n"] = n
    cst_params["beta_sustained"] = beta_sustained
    cst_params["beta_transient"] = beta_transient

    return cst_params
