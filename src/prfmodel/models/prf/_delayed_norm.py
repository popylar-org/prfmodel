"""Delayed normalization population receptive field models."""

import pandas as pd
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.regressors.base import BaseRegressors
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from ._gaussian import Gaussian2DPRFResponse
from ._stimulus_encoding import PRFStimulusEncoder
from .canonical import DelayedNormPRFModel


class DelayedNormGaussian2DPRFModel(DelayedNormPRFModel):
    r"""
    Delayed normalization pRF model with a 2D isotropic Gaussian response.

    Thin wrapper around :class:`DelayedNormPRFModel` that hardcodes
    :class:`~prfmodel.models.prf.Gaussian2DPRFResponse` as the pRF model.

    Parameters
    ----------
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=BaselineAmplitude
        Scaling model applied to R(t) after the nonlinear stage.
    %(model_regressors)s

    Notes
    -----
    Paper-recommended starting values (Fig. 2): ``n=2``, ``dispersion_normalization=0.1`` (τ₂),
    ``sigma_saturation=1`` (sigma), ``delay=0.05`` (τ₁), ``weight_deriv=0`` (w).

    Use :func:`init_delayed_norm_from_gaussian` to seed these from a fitted Gaussian model.

    Using the default impulse model, the following columns are expected in the
    :class:`pandas.DataFrame` passed as the ``parameters`` argument to :meth:`__call__`:

    .. list-table::
       :header-rows: 1
       :widths: 22 12 12 49

       * - Parameter
         - Model
         - Paper symbol
         - Description
       * - ``mu_x``
         - pRF
         - —
         - x-coordinate of the Gaussian centre.
       * - ``mu_y``
         - pRF
         - —
         - y-coordinate of the Gaussian centre.
       * - ``sigma``
         - pRF
         - —
         - Standard deviation of the isotropic Gaussian.
       * - ``delay``
         - h₁
         - τ₁ (default 0.05)
         - Peak time of the positive gamma component (in seconds).
       * - ``dispersion``
         - h₁
         - —
         - Rate parameter of the positive gamma component.
       * - ``undershoot``
         - h₁
         - —
         - Peak time of the negative gamma component (in seconds).
       * - ``u_dispersion``
         - h₁
         - —
         - Rate parameter of the negative gamma component.
       * - ``ratio``
         - h₁
         - —
         - Weight of the negative gamma component.
       * - ``weight_deriv``
         - h₁
         - w (default 0)
         - Weight of the derivative component.
       * - ``n``
         - --
         - n (default 2)
         - Exponent for the nonlinear stage.
       * - ``dispersion_normalization``
         - --
         - τ₂ (default 0.1)
         - Time constant of the exponential low-pass kernel h₂ (seconds).
       * - ``sigma_saturation``
         - --
         - sigma (default 1)
         - Semi-saturation constant.
       * - ``amplitude``
         - Scaling
         - —
         - Multiplicative output scale.
       * - ``baseline``
         - Scaling
         - —
         - Additive output constant.

    References
    ----------
    .. [1] Zhou J., Benson N.C., Kay K., Winawer J. (2019). Predicting neuronal dynamics with a
        delayed gain control model. *PLOS Computational Biology*, 15(9).
        https://doi.org/10.1371/journal.pcbi.1007484

    Examples
    --------
    Predict a model response for multiple units.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> model = DelayedNormGaussian2DPRFModel()
    >>> params = pd.DataFrame({
    ...     "mu_x": [0.0, 1.0],
    ...     "mu_y": [1.0, 0.0],
    ...     "sigma": [1.0, 1.5],
    ...     "delay": [0.05, 0.05],
    ...     "dispersion": [0.9, 0.9],
    ...     "undershoot": [12.0, 12.0],
    ...     "u_dispersion": [0.9, 0.9],
    ...     "ratio": [0.48, 0.48],
    ...     "weight_deriv": [0.0, 0.0],
    ...     "n": [2.0, 2.0],
    ...     "dispersion_normalization": [0.1, 0.1],
    ...     "sigma_saturation": [1.0, 1.0],
    ...     "amplitude": [1.0, 1.0],
    ...     "baseline": [0.0, 0.0],
    ... })
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (2, 170)

    """

    def __init__(
        self,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = BaselineAmplitude,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        super().__init__(
            prf_model=Gaussian2DPRFResponse(),
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )


def init_delayed_norm_from_gaussian(
    gaussian_params: pd.DataFrame,
    n: float = 2.0,
    dispersion_normalization: float = 0.1,
    sigma_saturation: float = 1.0,
) -> pd.DataFrame:
    """
    Initialize delayed normalization parameters from fitted Gaussian parameters.

    Converts the output of a fitted :class:`~prfmodel.models.prf.Gaussian2DPRFModel`
    into starting parameters for a :class:`DelayedNormGaussian2DPRFModel`, suitable
    for subsequent fitting with Stochastic Gradient Descent. All existing columns (pRF, impulse, and scaling parameters)
    pass through unchanged. The three DelayedNorm-specific parameters are appended with their
    default values.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a :class:`~prfmodel.models.prf.Gaussian2DPRFModel`.
    n : float, default=2.0
        Exponent for the nonlinear stage. Paper-recommended default.
    dispersion_normalization : float, default=0.1
        Time constant of the exponential decay kernel h₂ in seconds. Paper-recommended default.
    sigma_saturation : float, default=1.0
        Semi-saturation constant. Paper-recommended default.

    Returns
    -------
    pandas.DataFrame
        Copy of ``gaussian_params`` with three additional columns:
        ``n``, ``dispersion_normalization``, ``sigma_saturation``.

    Notes
    -----
    The paper also recommends setting ``delay=0.05`` (τ₁) and ``weight_deriv=0`` (w)
    in the impulse model. These are impulse parameters and should be set in the
    ``gaussian_params`` DataFrame before calling this function if desired.

    References
    ----------
    .. [1] Zhou J., Benson N.C., Kay K., Winawer J. (2019). Predicting neuronal dynamics with a
        delayed gain control model. *PLOS Computational Biology*, 15(9).
        https://doi.org/10.1371/journal.pcbi.1007484

    Examples
    --------
    >>> import pandas as pd
    >>> gaussian_params = pd.DataFrame({
    ...     "mu_x": [0.0, 1.0],
    ...     "mu_y": [0.0, -1.0],
    ...     "sigma": [1.0, 2.0],
    ...     "baseline": [0.0, 0.1],
    ...     "amplitude": [1.0, -1.0],
    ... })
    >>> dgn_params = init_delayed_norm_from_gaussian(gaussian_params)
    >>> print(sorted(dgn_params.columns.tolist()))
    ['amplitude', 'baseline', 'dispersion_normalization', 'mu_x', 'mu_y', 'n', 'sigma', 'sigma_saturation']

    """
    dgn_params = gaussian_params.copy()
    dgn_params["n"] = n
    dgn_params["dispersion_normalization"] = dispersion_normalization
    dgn_params["sigma_saturation"] = sigma_saturation
    return dgn_params
