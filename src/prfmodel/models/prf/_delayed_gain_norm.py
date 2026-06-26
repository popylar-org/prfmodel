"""Delayed gain normalization population receptive field models."""

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import cast
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _validate_regressors_argument
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from ._gaussian import Gaussian2DPRFModel

if TYPE_CHECKING:
    from prfmodel.models.base import BasePopulationResponse
    from prfmodel.models.base import BaseStimulusEncoder


class DelayedGainNormGaussian2DPRFModel(Gaussian2DPRFModel):
    r"""
    Two-dimensional isotropic Gaussian pRF model with delayed gain control normalization.

    Subclass of :class:`~prfmodel.models.prf.Gaussian2DPRFModel` that replaces the standard
    amplitude/baseline scaling with a delayed gain normalization stage computed inline.

    Parameters
    ----------
    %(model_impulse)s
    %(model_regressors)s
    duration : float, default=32.0
        Duration of the exponential decay kernel in seconds.
    resolution : float, default=1.0
        Seconds per frame.

    Notes
    -----
    The delayed gain normalization model follows the following steps [1]_:
      1. **Linear** — Gaussian pRF response encoded with the stimulus design, then convolved
        with the impulse response h₁ (parameters: ``delay``, ``weight_deriv``, etc.).
      2. **Nonlinear + Divide** — Delayed gain normalization applied to L(t):
        ``R(t) = |L(t)|^n / (sigma^n + |(L * h2)(t)|^n)``
        (parameters: ``n``, ``tau_2``, ``sigma_saturation``).
      3. **Output scaling** — ``amplitude * R(t) + baseline``.

    Paper-recommended starting values (Fig. 2): ``n=2``, ``tau_2=0.1`` (tau_2),
    ``sigma_saturation=1`` (sigma), ``delay=0.05`` (tau_1), ``weight_deriv=0`` (w).

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
         - Impulse
         - τ₁ (default 0.05)
         - Peak time of the positive gamma component (in seconds).
       * - ``dispersion``
         - Impulse
         - —
         - Rate parameter of the positive gamma component.
       * - ``undershoot``
         - Impulse
         - —
         - Peak time of the negative gamma component (in seconds).
       * - ``u_dispersion``
         - Impulse
         - —
         - Rate parameter of the negative gamma component.
       * - ``ratio``
         - Impulse
         - —
         - Weight of the negative gamma component.
       * - ``weight_deriv``
         - Impulse
         - w (default 0)
         - Weight of the derivative component.
       * - ``n``
         - DGN
         - n (default 2)
         - Exponent for the nonlinear stage (must be >= 1).
       * - ``tau_2``
         - DGN
         - τ₂ (default 0.1)
         - Time constant of the exponential low-pass kernel h₂ (seconds).
       * - ``sigma_saturation``
         - DGN
         - sigma (default 1)
         - Semi-saturation constant.
       * - ``amplitude``
         - DGN
         - —
         - Multiplicative output scale.
       * - ``baseline``
         - DGN
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
    >>> model = DelayedGainNormGaussian2DPRFModel()
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
    ...     "tau_2": [0.1, 0.1],
    ...     "sigma_saturation": [1.0, 1.0],
    ...     "amplitude": [1.0, 1.0],
    ...     "baseline": [0.0, 0.0],
    ... })
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (2, 200)

    """

    _DGN_PARAMETER_NAMES: ClassVar[list[str]] = ["n", "tau_2", "sigma_saturation", "amplitude", "baseline"]

    def __init__(
        self,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
        duration: float = 32.0,
        resolution: float = 1.0,
    ):
        super().__init__(
            impulse_model=impulse_model,
            scaling_model=None,
            regressors_model=regressors_model,
        )
        self.duration = duration
        self.resolution = resolution
        self._t_cached: Tensor | None = None

    @property
    def _frames(self) -> Tensor:
        """Cached time axis for the exponential kernel, shape (1, num_kernel_frames)."""
        if self._t_cached is None:
            num_kernel_frames = int(self.duration / self.resolution)
            self._t_cached = ops.expand_dims(
                ops.linspace(0.0, self.duration, num_kernel_frames),
                0,
            )
        return self._t_cached

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model (pRF + impulse + DGN)."""
        return super().parameter_names + self._DGN_PARAMETER_NAMES

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

        Parameters
        ----------
        %(stimulus_prf)s
        %(parameters)s

            - ``n`` : Exponent for the nonlinear stage. Must be >= 1 for all units.
            - ``tau_2`` : Time constant of the exponential decay kernel h₂ in seconds.
            - ``sigma_saturation`` : Semi-saturation constant.
            - ``amplitude`` : Multiplicative output scale.
            - ``baseline`` : Additive output constant.
        %(regressors_canonical)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        _validate_regressors_argument(self.models["regressors_model"], regressors)

        n_values = parameters["n"].to_numpy()
        if (n_values < 1.0).any():
            bad = n_values[n_values < 1.0].tolist()
            msg = f"All values of 'n' must be >= 1, but got: {bad}"
            raise ValueError(msg)

        # pRF response --> stimulus encoding
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters, dtype=dtype)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        # Impulse convolution (L(t))
        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        # DGN: R(t) = |L(t)|^n / (sigma^n + |(L * h2)(t)|^n)
        n = convert_parameters_to_tensor(parameters[["n"]], dtype=dtype)
        tau_2 = convert_parameters_to_tensor(parameters[["tau_2"]], dtype=dtype)
        sigma_saturation = convert_parameters_to_tensor(parameters[["sigma_saturation"]], dtype=dtype)
        amplitude = convert_parameters_to_tensor(parameters[["amplitude"]], dtype=dtype)
        baseline = convert_parameters_to_tensor(parameters[["baseline"]], dtype=dtype)

        r_ln = ops.power(ops.abs(response), n)

        t = ops.cast(self._frames, dtype=dtype)
        kernel = ops.exp(-t / tau_2)

        g_t = convolve_prf_impulse_response(response, kernel)
        denominator = ops.power(sigma_saturation, n) + ops.power(ops.abs(g_t), n)
        response = amplitude * (r_ln / denominator) + baseline

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model(regressors, parameters, dtype=dtype)

        return response


def init_delayed_gain_norm_from_gaussian(  # noqa: PLR0913
    gaussian_params: pd.DataFrame,
    n: float = 2.0,
    tau_2: float = 0.1,
    sigma_saturation: float = 1.0,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> pd.DataFrame:
    """
    Initialize delayed gain normalization parameters from fitted Gaussian parameters.

    Converts the output of a fitted :class:`~prfmodel.models.prf.Gaussian2DPRFModel`
    into starting parameters for a :class:`DelayedGainNormGaussian2DPRFModel`, suitable
    for subsequent SGD. All existing columns (pRF and impulse parameters) pass through
    unchanged. The five DGN-specific parameters are appended with their default values.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a :class:`~prfmodel.models.prf.Gaussian2DPRFModel`.
    n : float, default=2.0
        Exponent for the nonlinear stage. Paper-recommended default.
    tau_2 : float, default=0.1
        Time constant of the exponential decay kernel h₂ in seconds. Paper-recommended default.
    sigma_saturation : float, default=1.0
        Semi-saturation constant. Paper-recommended default.
    amplitude : float, default=1.0
        Multiplicative output scale.
    baseline : float, default=0.0
        Additive output constant.

    Returns
    -------
    pandas.DataFrame
        Copy of ``gaussian_params`` with five additional columns:
        ``n``, ``tau_2``, ``sigma_saturation``, ``amplitude``, ``baseline``.

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
    ['amplitude', 'baseline', 'mu_x', 'mu_y', 'n', 'sigma', 'sigma_saturation', 'tau_2']

    """
    dgn_params = gaussian_params.copy()
    dgn_params["n"] = n
    dgn_params["tau_2"] = tau_2
    dgn_params["sigma_saturation"] = sigma_saturation
    dgn_params["amplitude"] = amplitude
    dgn_params["baseline"] = baseline
    return dgn_params
