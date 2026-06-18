"""Canonical population receptive field (pRF) models.

This module contains models that combine multiple exchangeable submodels in a way that is considered "canonical".

"""

from abc import abstractmethod
from typing import ClassVar
from typing import cast
import pandas as pd
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
from prfmodel.scaling import Baseline
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from ._stimulus_encoding import PRFStimulusEncoder


class CanonicalPRFModel(BaseCanonical[PRFStimulus]):
    """
    Canonical population receptive field (pRF) model.

    This class combines a pRF response, impulse, and scaling model.

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

    1. The population receptive field response model makes a prediction for the stimulus grid.
    2. The encoding model encodes the response with the stimulus design.
    3. The impulse model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The scaling model modifies the convolved response.
    6. The regressors model (optional) adds a linear combination of fixed regressors to the scaled response.

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
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a simple population receptive field model response to a stimulus.

        Parameters
        ----------
        %(stimulus_prf)s
        %(parameters)s
        %(regressors_canonical)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        _validate_regressors_argument(self.models["regressors_model"], regressors)

        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters, dtype=dtype)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["scaling_model"] is not None:
            temporal_model = cast("BaseScaling", self.models["scaling_model"])
            response = temporal_model(response, parameters, dtype=dtype)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model(regressors, parameters, dtype=dtype)

        return response


class _BaseDualPRFModel(BaseCanonical[PRFStimulus]):
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
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        suffix: str,
        dtype: str,
    ) -> Tensor:
        """Predict a single encoded pRF model response.

        Shared parameters are taken as-is; non-shared parameters are read from the ``{param}_{suffix}`` columns.
        Only the columns the pRF model consumes are gathered, avoiding a copy of the full parameter frame.

        """
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        shared = set(self.shared_params)

        params_single = pd.DataFrame(
            {
                param: parameters[param if param in shared else f"{param}_{suffix}"]
                for param in prf_model.parameter_names
            },
        )

        response = prf_model(stimulus, params_single, dtype=dtype)

        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])

        return encoding_model(stimulus, response, parameters, dtype=dtype)

    @abstractmethod
    def _combine_responses(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str,
    ) -> Tensor:
        """Combine the two encoded pRF responses into a single response before the impulse/scaling tail."""

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the combined model response to a stimulus.

        Parameters
        ----------
        %(stimulus_prf)s
        %(parameters)s
        %(regressors_canonical)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        _validate_regressors_argument(self.models["regressors_model"], regressors)

        response = self._combine_responses(stimulus, parameters, dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["scaling_model"] is not None:
            scaling_model = cast("BaseScaling", self.models["scaling_model"])
            response = scaling_model(response, parameters, dtype=dtype)

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model(regressors, parameters, dtype=dtype)

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
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str,
    ) -> Tensor:
        amplitude_center = convert_parameters_to_tensor(parameters[["amplitude_center"]], dtype=dtype)
        amplitude_surround = convert_parameters_to_tensor(parameters[["amplitude_surround"]], dtype=dtype)

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

    def _combine_responses(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str,
    ) -> Tensor:
        a = convert_parameters_to_tensor(parameters[["amplitude_activation"]], dtype=dtype)
        c = convert_parameters_to_tensor(parameters[["amplitude_normalization"]], dtype=dtype)

        b = convert_parameters_to_tensor(parameters[["baseline_activation"]], dtype=dtype)
        d = convert_parameters_to_tensor(parameters[["baseline_normalization"]], dtype=dtype)

        response_activation = a * self._predict_single_response(stimulus, parameters, "activation", dtype) + b
        response_normalization = c * self._predict_single_response(stimulus, parameters, "normalization", dtype) + d

        return response_activation / response_normalization - b / d
