"""Canonical population receptive field (pRF) models.

This module contains models that combine multiple exchangeable submodels in a way that is considered "canonical".

"""

import warnings
from abc import abstractmethod
from typing import ClassVar
from typing import cast
from keras import ops
from prfmodel._docstring import doc
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import SustainedImpulse
from prfmodel.impulse import TransientImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BasePopulationResponse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _normalize_regressors_model
from prfmodel.scaling import Baseline
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from prfmodel.stimuli import PRFStimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol
from prfmodel.utils import ParamsDict
from prfmodel.utils import normalize_response
from ._stimulus_encoding import PRFStimulusEncoder


class CanonicalPRFModel(BaseCanonical[PRFStimulus, PRFStimulusTensors]):
    """
    Canonical population receptive field (pRF) model.

    This class combines a pRF response, impulse, scaling, and regressors model.

    Parameters
    ----------
    %(model_prf)s
    %(model_encoding_prf)s
    %(model_impulse)s
    %(model_scaling)s
    %(model_regressors)s

    Notes
    -----
    The canonical model follows the following steps:

    1. The pRF response model makes a prediction for the stimulus grid.
    2. The encoding model encodes the response with the stimulus design.
    3. The encoded response is convolved with an impulse response (optional).
    4. The scaling model modifies the convolved response (optional).
    5. The regressors model adds a linear combination of fixed regressors to the scaled response (optional).

    """

    @doc
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

    @doc
    def call(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        regressors: ParamsDict | None = None,
    ) -> Tensor:
        """
        Predict the model response to a stimulus.

        Parameters
        ----------
        %(stimulus_prf_tensors)
        %(parameters_tensors)
        %(regressors_tensors)

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = parameters.dtype

        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model.call(stimulus, parameters)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model.call(stimulus, response, parameters)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model.call(parameters)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["scaling_model"] is not None:
            temporal_model = cast("BaseScaling", self.models["scaling_model"])
            response = temporal_model.call(response, parameters)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model.call(regressors, parameters)

        return response


class _BaseDualPRFModel(BaseCanonical[PRFStimulus, PRFStimulusTensors]):
    """Shared base for dual pRF models that combine two encoded pRF responses.

    Concrete subclasses run two pRF responses through stimulus encoding (via the shared
    :meth:`_predict_single_response`), combine them in :meth:`_combine_responses`, and then share the common
    impulse-convolution, scaling, and regressor tail implemented in :meth:`__call__`.

    Subclasses configure the parameter contract with two class attributes: ``_response_suffixes`` (the two
    suffixes appended to non-shared pRF parameters) and ``_combine_param_names`` (the extra parameter columns that
    :meth:`_combine_responses` consumes).

    """

    _response_suffixes: ClassVar[tuple[str, str]]
    _combine_param_names: ClassVar[tuple[str, ...]]

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        prf_model: BasePopulationResponse,
        shared_params: list[str] | None = None,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        shared_params = shared_params or []

        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        regressors_model = _normalize_regressors_model(regressors_model)

        invalid = [p for p in shared_params if p not in prf_model.parameter_names]

        if invalid:
            msg = f"Shared parameters {invalid} not found in 'prf_model.parameter_names' {prf_model.parameter_names}"
            raise ValueError(msg)

        self.shared_params = list(shared_params)

        non_shared = [p for p in prf_model.parameter_names if p not in self.shared_params]
        collision = {f"{p}_{suffix}" for p in non_shared for suffix in self._response_suffixes} & set(
            self._combine_param_names,
        )
        if collision:
            msg = (
                f"Suffixed pRF parameter(s) {sorted(collision)} collide with the combination parameters "
                f"{list(self._combine_param_names)}; rename the pRF parameter or add it to 'shared_params'."
            )
            raise ValueError(msg)

        super().__init__(
            prf_model=prf_model,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Shared response model parameters appear once (no suffix). Non-shared parameters are suffixed with the two
        response suffixes (e.g. ``_center``/``_surround``). The combination, encoding, impulse and scaling
        parameters are appended.

        """
        suffixed = self._suffixed_prf_param_names()
        combine = list(self._combine_param_names)
        param_names = [*suffixed, *combine]

        for key, model in self.models.items():
            if key != "prf_model" and model is not None:
                param_names.extend(model.parameter_names)

        return list(dict.fromkeys(param_names))

    def _suffixed_prf_param_names(self) -> list[str]:
        """Return pRF parameter names with shared params as-is and non-shared params expanded for both responses."""
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        shared = set(self.shared_params)
        first, second = self._response_suffixes

        names = [p if p in shared else f"{p}_{first}" for p in prf_model.parameter_names]
        names.extend(f"{p}_{second}" for p in prf_model.parameter_names if p not in shared)
        return names

    def _predict_single_response(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        suffix: str,
        dtype: str,
    ) -> Tensor:
        """Predict a single encoded pRF model response.

        Shared parameters are taken as-is; non-shared parameters are read from the ``{param}_{suffix}`` columns.
        Only the columns the pRF model consumes are gathered, avoiding a copy of the full parameter frame.

        Gathered into a fresh :class:`~prfmodel.utils.ParamsDict` because the pRF submodel expects its own
        unsuffixed parameter names.

        """
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        shared = set(self.shared_params)

        params_single = ParamsDict(
            {
                param: parameters[param if param in shared else f"{param}_{suffix}"]
                for param in prf_model.parameter_names
            },
            dtype=dtype,
        )

        response = prf_model.call(stimulus, params_single)

        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])

        return encoding_model.call(stimulus, response, parameters)

    @abstractmethod
    def _combine_responses(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        dtype: str,
    ) -> Tensor:
        """Combine the two encoded pRF responses into a single response before the impulse/scaling tail."""

    @doc
    def call(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        regressors: ParamsDict | None = None,
    ) -> Tensor:
        """
        Predict the combined model response to a stimulus.

        Parameters
        ----------
        %(stimulus_prf_tensors)
        %(parameters_tensors)
        %(regressors_tensors)

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = parameters.dtype

        response = self._combine_responses(stimulus, parameters, dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model.call(parameters)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["scaling_model"] is not None:
            scaling_model = cast("BaseScaling", self.models["scaling_model"])
            response = scaling_model.call(response, parameters)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model.call(regressors, parameters)

        return response


class CenterSurroundPRFModel(_BaseDualPRFModel):
    """
    Center-surround population receptive field (pRF) model.

    This class combines the difference between a center pRF response and a surround pRF response with an
    impulse and scaling model. Both the center and surround response come from the same model class, but their
    parameters can differ.

    Parameters
    ----------
    %(model_prf)s
    shared_params : list of str, optional
        Names of parameters that are shared between the two pRF response models. All names must appear in
        ``prf_model.parameter_names``.
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline, optional
        A scaling model class or instance. Model classes will be instantiated during initialization.
        The default creates a :class:`~prfmodel.scaling.Baseline` instance.
    %(model_regressors)s

    Notes
    -----
    The center-surround model follows these steps:

    1. The two pRF response models make predictions for the stimulus grid.
    2. The encoding model encodes the responses with the stimulus design.
    3. The encoded responses are scaled with separate amplitudes. The surround response is subtracted from the
       center response yielding the combined response.
    4. The combined response is convolved with an impulse response (optional).
    5. The scaling model modifies the convolved response (optional).
    6. The regressors model adds a linear combination of fixed regressors to the scaled response (optional).

    """

    _response_suffixes: ClassVar[tuple[str, str]] = ("center", "surround")
    _combine_param_names: ClassVar[tuple[str, ...]] = ("amplitude_center", "amplitude_surround")

    def _combine_responses(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        dtype: str,
    ) -> Tensor:
        amplitude_center = parameters[["amplitude_center"]]
        amplitude_surround = parameters[["amplitude_surround"]]

        response_center = amplitude_center * self._predict_single_response(stimulus, parameters, "center", dtype)
        response_surround = amplitude_surround * self._predict_single_response(stimulus, parameters, "surround", dtype)

        return response_center - response_surround


class DivNormPRFModel(_BaseDualPRFModel):
    r"""
    Divisive normalization population receptive field (pRF) model.

    This class performs divisive normalization between an activation (numerator) and a normalization (denominator)
    pRF response and combines them with an impulse and scaling model. Both responses come from the same model class,
    but their parameters can differ.

    Parameters
    ----------
    %(model_prf)s
    shared_params : list of str, optional
        Names of pRF parameters that are shared between the two responses. All names must appear in
        ``prf_model.parameter_names``.
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline, optional
        A scaling model class or instance. Model classes will be instantiated during initialization.
        The default creates a :class:`~prfmodel.scaling.Baseline` instance.
    %(model_regressors)s
    min_response : float, default=1e-10
        Lower bound applied to ``baseline_normalization`` before it is used.
        Keeps the ``b / d`` offset term finite when ``baseline_normalization`` is zero.

    Notes
    -----
    The divisive normalization model follows these steps

    1. The two pRF response models make predictions for the stimulus grid.
    2. The encoding model encodes the responses with the stimulus design.
    3. The two encoded responses are combined through divisive normalization.
    4. The combined response is convolved with an impulse response (optional).
    5. The scaling model modifies the convolved response (optional).
    6. The regressors model adds a linear combination of fixed regressors to the scaled response (optional).

    """

    _response_suffixes: ClassVar[tuple[str, str]] = ("activation", "normalization")
    _combine_param_names: ClassVar[tuple[str, ...]] = (
        "amplitude_activation",
        "amplitude_normalization",
        "baseline_activation",
        "baseline_normalization",
    )

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        prf_model: BasePopulationResponse,
        shared_params: list[str] | None = None,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
        min_baseline_normalization: float = 1e-10,
    ):
        self.min_baseline_normalization = min_baseline_normalization

        super().__init__(
            prf_model=prf_model,
            shared_params=shared_params,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    def _combine_responses(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        dtype: str,
    ) -> Tensor:
        a = parameters[["amplitude_activation"]]
        c = parameters[["amplitude_normalization"]]

        b = parameters[["baseline_activation"]]
        d = parameters[["baseline_normalization"]]
        d = ops.maximum(d, self.min_baseline_normalization)

        response_activation = a * self._predict_single_response(stimulus, parameters, "activation", dtype) + b
        response_normalization = c * self._predict_single_response(stimulus, parameters, "normalization", dtype) + d

        return response_activation / response_normalization - b / d


class DelayedNormPRFModel(BaseCanonical[PRFStimulus, PRFStimulusTensors]):
    r"""
    Delayed gain normalization population receptive field (pRF) model.

    Combines a pRF response model, stimulus encoding, and an impulse response (h₁) with an
    inline delayed normalization stage (h₂ = exponential decay) to form a complete DGN model.
    The computation and all DGN-specific parameters (``n``, ``dispersion_normalization``, ``sigma_saturation``,
    ``amplitude``, ``baseline``) live in this class; pRF-specific parameters come from
    ``prf_model``.

    Parameters
    ----------
    %(model_prf)s
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
    def call(
        self,
        stimulus: PRFStimulusTensors,
        parameters: ParamsDict,
        regressors: ParamsDict | None = None,
    ) -> Tensor:
        """
        Predict the delayed gain normalization model response.

        Parameters
        ----------
        %(stimulus_prf_tensors)
        %(parameters_tensors)
        %(regressors_tensors)

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = parameters.dtype

        # pRF response + stimulus encoding
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model.call(stimulus, parameters)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model.call(stimulus, response, parameters)

        impulse_model = cast("BaseImpulse", self.models["impulse_model"])

        # h₁ convolution → L(t)
        if impulse_model is not None:
            impulse_response = impulse_model.call(parameters)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        # DGN parameters
        n = parameters[["n"]]
        dispersion_normalization = parameters[["dispersion_normalization"]]
        sigma_saturation = parameters[["sigma_saturation"]]

        # h₂ kernel → g(t) = L * h₂
        # Without an impulse model there is no time axis to build h₂ on, so the normalization stage is
        # deliberately skipped and the model reduces to a static saturating nonlinearity.
        if impulse_model is not None:
            t = impulse_model.get_frames(dtype)
            kernel = normalize_response(ops.exp(-t / dispersion_normalization), impulse_model.norm)
            g_t = convolve_prf_impulse_response(response, kernel, dtype=dtype)
        else:
            g_t = response

        # R(t) = |L(t)|ⁿ / (sigmaⁿ + |g(t)|ⁿ)
        # The absolute values follow Zhou et al. (2019) and are intentional: they keep the power law defined for
        # non-integer n, at the cost of rectifying negative deflections of L(t) and g(t).
        r_ln = ops.power(ops.abs(response), n)
        denominator = ops.power(sigma_saturation, n) + ops.power(ops.abs(g_t), n)
        response = r_ln / denominator

        if self.models["scaling_model"] is not None:
            scaling_model = cast("BaseScaling", self.models["scaling_model"])
            response = scaling_model.call(response, parameters)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model.call(regressors, parameters)

        return response


class ResolutionMismatchWarning(UserWarning):
    """Warning for when submodels of a canonical model are sampled on time axes of different resolutions."""


class CSTPRFModel(BaseCanonical[PRFStimulus]):
    r"""
    Compressive spatiotemporal (CST) population receptive field (pRF) model.

    Combines a single pRF response model with three temporal channels that share one spatial receptive field:
    a sustained channel, an on-transient channel, and an off-transient channel. Each channel response is
    rectified and compressed, the channels are combined with separate weights, and the result is convolved with
    an impulse response.

    The compression and weighting parameters (``n``, ``amplitude_sustained``, ``amplitude_transient``) are owned by this
    class. Every other parameter is contributed by a submodel and changes when that submodel is replaced:

    - ``sustained_model`` and ``transient_model`` supply the channel timing parameter (``time_to_peak`` for the
      default :class:`~prfmodel.impulse.SustainedImpulse` and :class:`~prfmodel.impulse.TransientImpulse`).
    - ``prf_model`` supplies the spatial parameters (``mu_y``, ``mu_x`` and ``sigma`` for
      :class:`~prfmodel.models.prf.Gaussian2DPRFResponse`).
    - ``impulse_model`` and ``scaling_model`` supply the rest (``weight_deriv`` and ``baseline`` for the
      defaults; the other impulse parameters are covered by its default parameter set).

    Parameters
    ----------
    %(model_prf)s
    %(model_encoding_prf)s
    sustained_model : BaseImpulse or type, default=SustainedImpulse
        Temporal channel model producing the sustained response h₁.
    transient_model : BaseImpulse or type, default=TransientImpulse
        Temporal channel model producing the on-transient response h₂. The off-transient h₃ is its negation.
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline
        Scaling model applied after the channels are combined. The channel weights already provide amplitudes,
        so the default adds only an offset.
    %(model_regressors)s
    min_response : float, default=1e-10
        Lower bound applied by the rectifier before compression. A small positive value rather than zero keeps
        gradients of the power law finite when ``n < 1``.

    Notes
    -----
    The compressive spatiotemporal model follows [1]_:

      1. **Spatial** — the pRF response is encoded with the stimulus design, giving the linear response that the
         reference writes as ``I(X, Y, t) · RF(X, Y)``.
      2. **Temporal** — that response is convolved with each channel: ``r_i(t) = h_i(t) * [I . RF]`` for
         ``i = 1, 2, 3``, where ``h_3 = -h_2``.
      3. **Nonlinear Compression** — ``pᵢ(t) = [ReLU(rᵢ(t))]ⁿ`` with ``0.1 ≤ n ≤ 1``.
      4. **Output** — each weighted channel group is convolved with the impulse response HRF and summed:
         ``β_sus * (p₁(t) * HRF) + β_tran * ([p₂(t) + p₃(t)] * HRF)``.

    ``time_to_peak`` is shared by both channel models, so it appears once in :attr:`parameter_names`, matching
    the reference where the temporal and compression parameters are identical across channels.

    References
    ----------
    .. [1] Kim, I., Kupers, E. R., Lerma-Usabiaga, G., & Grill-Spector, K. (2024). Characterizing spatiotemporal
        population receptive fields in human visual cortex with fMRI. *The Journal of Neuroscience*, 44(2),
        e0803232023. https://doi.org/10.1523/JNEUROSCI.0803-23.2023

    """

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        prf_model: BasePopulationResponse,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        sustained_model: BaseImpulse | type[BaseImpulse] = SustainedImpulse,
        transient_model: BaseImpulse | type[BaseImpulse] = TransientImpulse,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
        min_response: float = 1e-10,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if isinstance(sustained_model, type):
            sustained_model = sustained_model()

        if isinstance(transient_model, type):
            transient_model = transient_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        regressors_model = _normalize_regressors_model(regressors_model)

        _check_channel_models(sustained_model, transient_model)
        _check_channel_resolution(sustained_model, transient_model, impulse_model)

        self.min_response = min_response

        super().__init__(
            prf_model=prf_model,
            encoding_model=encoding_model,
            sustained_model=sustained_model,
            transient_model=transient_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model (pRF + encoding + channels + CST + impulse + scaling)."""
        names: list[str] = []

        # These submodels cannot be None (the channel models are checked in __init__), unlike those below
        for key, model in self.models.items():
            if key in ("prf_model", "encoding_model", "sustained_model", "transient_model"):
                names.extend(cast("ModelProtocol", model).parameter_names)

        names.extend(["n", "amplitude_sustained", "amplitude_transient"])

        for key, model in self.models.items():
            if key in ("impulse_model", "scaling_model", "regressors_model") and model is not None:
                names.extend(model.parameter_names)

        return list(dict.fromkeys(names))

    def _rectify_and_compress(self, response: Tensor, exponent: Tensor) -> Tensor:
        """Rectify a channel response and apply the compressive power law.

        The rectifier floors at :attr:`min_response` rather than zero so that the gradient of the power law
        stays finite for ``n < 1``, matching :class:`~prfmodel.models.compression.CompressiveEncoder`.

        """
        return ops.power(ops.maximum(response, self.min_response), exponent)

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the compressive spatiotemporal model response to a stimulus.

        Parameters
        ----------
        %(stimulus_prf)s
        %(parameters)s
        %(regressors_canonical)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        Raises
        ------
        %(raises_missing_parameters)s

        """
        self._check_parameters(parameters)
        dtype = get_dtype(dtype)
        _validate_regressors_argument(self.models["regressors_model"], regressors)

        # Spatial stage: pRF response encoded with the stimulus design
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters, dtype=dtype)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        # Temporal stage: one convolution per channel. The off-transient is the negated on-transient.
        sustained_model = cast("BaseImpulse", self.models["sustained_model"])
        transient_model = cast("BaseImpulse", self.models["transient_model"])

        response_sustained = convolve_prf_impulse_response(response, sustained_model(parameters, dtype), dtype=dtype)
        response_transient = convolve_prf_impulse_response(response, transient_model(parameters, dtype), dtype=dtype)

        # Nonlinear stage
        n = convert_parameters_to_tensor(parameters[["n"]], dtype=dtype)
        sustained = self._rectify_and_compress(response_sustained, n)
        transient_on = self._rectify_and_compress(response_transient, n)
        transient_off = self._rectify_and_compress(-response_transient, n)

        # Weighted combination of the sustained and transient channels, each convolved with the impulse response
        amplitude_sustained = convert_parameters_to_tensor(parameters[["amplitude_sustained"]], dtype=dtype)
        amplitude_transient = convert_parameters_to_tensor(parameters[["amplitude_transient"]], dtype=dtype)

        sustained = amplitude_sustained * sustained
        transient = amplitude_transient * (transient_on + transient_off)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)

            sustained = convolve_prf_impulse_response(sustained, impulse_response, dtype=dtype)
            transient = convolve_prf_impulse_response(transient, impulse_response, dtype=dtype)

        response = sustained + transient

        if self.models["scaling_model"] is not None:
            scaling_model = cast("BaseScaling", self.models["scaling_model"])
            response = scaling_model(response, parameters, dtype=dtype)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model(regressors, parameters, dtype=dtype)

        return response


def _check_channel_models(sustained_model: BaseImpulse | None, transient_model: BaseImpulse | None) -> None:
    """Raise if a temporal channel model is missing.

    The compressive spatiotemporal model is defined by its three channels, so omitting one has no meaning. A
    channel is silenced by setting its weight to zero, which keeps the model differentiable in that weight.

    """
    for name, model in (("sustained_model", sustained_model), ("transient_model", transient_model)):
        if model is None:
            msg = f"'{name}' is required; set its channel weight to zero to silence a channel instead"
            raise ValueError(msg)


def _check_channel_resolution(
    sustained_model: BaseImpulse,
    transient_model: BaseImpulse,
    impulse_model: BaseImpulse | None,
) -> None:
    """Warn if the channel models and the impulse model are sampled at different resolutions.

    :func:`~prfmodel.impulse.convolve_prf_impulse_response` requires its inputs to be sampled at the same rate
    but does not check it, so a mismatch yields a silently wrong prediction rather than an error. This is a
    warning rather than an error because a deliberate mismatch is meaningful: a finer neural time axis than the
    measurement axis is exactly how the reference computes its channels.

    """
    resolutions = {"sustained_model": sustained_model.resolution, "transient_model": transient_model.resolution}

    if impulse_model is not None:
        resolutions["impulse_model"] = impulse_model.resolution

    if len(set(resolutions.values())) > 1:
        msg = (
            f"Submodels are sampled at different resolutions ({resolutions}). They are convolved with one "
            f"another, so predictions are only meaningful when their time axes agree."
        )
        warnings.warn(message=msg, category=ResolutionMismatchWarning, stacklevel=3)
