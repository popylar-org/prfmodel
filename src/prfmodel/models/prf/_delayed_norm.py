"""Delayed gain normalization population receptive field models."""

from typing import cast
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BasePopulationResponse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _normalize_regressors_model
from prfmodel.regressors.base import _validate_regressors_argument
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from ._gaussian import Gaussian2DPRFResponse
from ._stimulus_encoding import PRFStimulusEncoder


class DelayedNormPRFModel(BaseCanonical[PRFStimulus]):
    r"""
    Delayed gain normalization population receptive field model.

    Combines a pRF response model, stimulus encoding, and an impulse response (h₁) with an
    inline delayed normalization stage (h₂ = exponential decay) to form a complete DGN model.
    The computation and all DGN-specific parameters (``n``, ``dispersion_normalization``, ``sigma_saturation``,
    ``amplitude``, ``baseline``) live in this class; pRF-specific parameters come from
    ``prf_model``.

    Parameters
    ----------
    prf_model : BasePopulationResponse
        Population receptive field response model.  Determines which spatial parameters
        (e.g., ``mu_x``, ``mu_y``, ``sigma``) are expected in ``parameters``.
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=BaselineAmplitude
        Scaling model applied to R(t) after the nonlinear stage.  Model classes are
        instantiated during initialisation.  Set to ``None`` to return R(t) unscaled.
    %(model_regressors)s

    Notes
    -----
    The delayed gain normalization model follows [1]_:

      1. **Linear** — pRF response encoded with the stimulus design, then convolved with
         the impulse response h₁ to produce L(t).
      2. **Normalization** — L(t) is convolved with h₂ = exp(-t/τ₂) to produce g(t).
      3. **Nonlinear** — ``R(t) = |L(t)|ⁿ / (sigmaⁿ + |g(t)|ⁿ)``.
      4. **Output** — ``amplitude * R(t) + baseline``.

    Paper-recommended starting values (Fig. 2): ``n=2``, ``dispersion_normalization=0.1``,
    ``sigma_saturation=1``, ``delay=0.05`` (τ₁), ``weight_deriv=0``.

    References
    ----------
    .. [1] Zhou J., Benson N.C., Kay K., Winawer J. (2019). Predicting neuronal dynamics with a
        delayed gain control model. *PLOS Computational Biology*, 15(9).
        https://doi.org/10.1371/journal.pcbi.1007484
    """

    def __init__(
        self,
        prf_model: BasePopulationResponse,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = BaselineAmplitude,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        regressors_model = _normalize_regressors_model(regressors_model)

        super().__init__(
            prf_model=prf_model,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model (pRF + h₁ impulse + DGN + scaling)."""
        names: list[str] = []
        for key, model in self.models.items():
            if key != "scaling_model" and model is not None:
                names.extend(model.parameter_names)
        names.extend(["n", "dispersion_normalization", "sigma_saturation"])
        if self.models["scaling_model"] is not None:
            names.extend(self.models["scaling_model"].parameter_names)
        return list(dict.fromkeys(names))

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the delayed gain normalization model response.

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        _validate_regressors_argument(self.models["regressors_model"], regressors)

        # pRF response + stimulus encoding
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters, dtype=dtype)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        impulse_model = cast("BaseImpulse", self.models["impulse_model"])

        # h₁ convolution → L(t)
        if impulse_model is not None:
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        # DGN parameters
        n = convert_parameters_to_tensor(parameters[["n"]], dtype=dtype)
        dispersion_normalization = convert_parameters_to_tensor(parameters[["dispersion_normalization"]], dtype=dtype)
        sigma_saturation = convert_parameters_to_tensor(parameters[["sigma_saturation"]], dtype=dtype)

        # h₂ kernel → g(t) = L * h₂
        if impulse_model is not None:
            t = ops.cast(impulse_model.frames, dtype=dtype)
            kernel = ops.exp(-t / dispersion_normalization)
            g_t = convolve_prf_impulse_response(response, kernel)
        else:
            g_t = response

        # R(t) = |L(t)|ⁿ / (sigmaⁿ + |g(t)|ⁿ)
        r_ln = ops.power(ops.abs(response), n)
        denominator = ops.power(sigma_saturation, n) + ops.power(ops.abs(g_t), n)
        response = r_ln / denominator

        if self.models["scaling_model"] is not None:
            scaling_model = cast("BaseScaling", self.models["scaling_model"])
            response = scaling_model(response, parameters, dtype=dtype)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model(regressors, parameters, dtype=dtype)

        return response


class DelayedNormGaussian2DPRFModel(DelayedNormPRFModel):
    r"""
    Delayed gain normalization pRF model with a 2D isotropic Gaussian response.

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

    Use :func:`init_delayed_gain_norm_from_gaussian` to seed these from a fitted Gaussian model.

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
         - DGN
         - n (default 2)
         - Exponent for the nonlinear stage (must be >= 1).
       * - ``dispersion_normalization``
         - DGN
         - τ₂ (default 0.1)
         - Time constant of the exponential low-pass kernel h₂ (seconds).
       * - ``sigma_saturation``
         - DGN
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
    (2, 200)

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


def init_delayed_gain_norm_from_gaussian(
    gaussian_params: pd.DataFrame,
    n: float = 2.0,
    dispersion_normalization: float = 0.1,
    sigma_saturation: float = 1.0,
) -> pd.DataFrame:
    """
    Initialize delayed gain normalization parameters from fitted Gaussian parameters.

    Converts the output of a fitted :class:`~prfmodel.models.prf.Gaussian2DPRFModel`
    into starting parameters for a :class:`DelayedNormGaussian2DPRFModel`, suitable
    for subsequent SGD. All existing columns (pRF, impulse, and scaling parameters)
    pass through unchanged. The three DGN-specific parameters are appended with their
    default values.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a :class:`~prfmodel.models.prf.Gaussian2DPRFModel`.
        Must already contain ``amplitude`` and ``baseline`` columns (from the scaling model).
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
    >>> dgn_params = init_delayed_gain_norm_from_gaussian(gaussian_params)
    >>> print(sorted(dgn_params.columns.tolist()))
    ['amplitude', 'baseline', 'dispersion_normalization', 'mu_x', 'mu_y', 'n', 'sigma', 'sigma_saturation']

    """
    dgn_params = gaussian_params.copy()
    dgn_params["n"] = n
    dgn_params["dispersion_normalization"] = dispersion_normalization
    dgn_params["sigma_saturation"] = sigma_saturation
    return dgn_params
