"""Canonical population receptive field (pRF) models.

This module contains models that combine multiple exchangeable submodels in a way that is considered "canonical".

"""

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
from prfmodel.regressors import RegressorsList
from prfmodel.regressors.base import BaseRegressors
from prfmodel.scaling import Baseline
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling import DoGAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from ._stimulus_encoding import PRFStimulusEncoder


def _normalize_regressors_model(
    regressors_model: BaseRegressors | list[BaseRegressors] | None,
) -> BaseRegressors | None:
    if isinstance(regressors_model, list):
        return RegressorsList(regressors_model)
    return regressors_model


def _validate_regressors_argument(
    regressors_model: object | None,
    regressors: pd.DataFrame | None,
) -> None:
    if regressors_model is None and regressors is not None:
        msg = "'regressors' was provided but 'regressors_model' is not configured on this model"
        raise ValueError(msg)
    if regressors_model is not None and regressors is None:
        msg = "'regressors' must be provided when 'regressors_model' is configured on this model"
        raise ValueError(msg)


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


class CenterSurroundPRFModel(BaseCanonical[PRFStimulus]):
    """
    Center-surround composite population receptive field model.

    This is a generic class that runs two PRF responses through stimulus encoding and impulse response convolution
    independently, then combines them via a temporal model:
    y(t) = p1(t) * amplitude_center + p2(t) * amplitude_surround + baseline

    The two responses differ in the values of ``change_params``. Each parameter
    ``p`` in that list is split into ``{p}_center`` and ``{p}_surround``.

    Parameters
    ----------
    %(model_prf)s
    %(model_encoding_prf)s
    %(model_impulse)s
    %(model_scaling)s
    %(model_regressors)s
    change_params : list[str], default=["sigma"]
        Names of the parameters that differ between the center and surround responses.
        All entries must be present in ``prf_model.parameter_names``.

    Notes
    -----
    The center-surround composite model follows these steps:

    1. Two PRF response predictions are computed, one using ``{p}_center`` and one
       using ``{p}_sorround`` for each ``p`` in ``change_params``.
    2. The encoding model encodes each response with the stimulus design.
    3. Each encoded response is optionally convolved with an impulse response.
    4. The two responses are stacked and combined by the scaling model.
    5. The regressors model (optional) adds a linear combination of fixed regressors to the combined response.

    """

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        prf_model: BasePopulationResponse,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = DoGAmplitude,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
        change_params: list[str] | None = None,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        regressors_model = _normalize_regressors_model(regressors_model)

        if change_params is None:
            change_params = ["sigma"]

        invalid = [p for p in change_params if p not in prf_model.parameter_names]
        if invalid:
            msg = (
                f"CenterSurroundPRFModel: parameter(s) {invalid} not found in "
                f"prf_model.parameter_names {prf_model.parameter_names}"
            )
            raise ValueError(msg)

        self._change_params = change_params

        super().__init__(
            prf_model=prf_model,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters used by the model."""
        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        prf_params = prf_model.parameter_names.copy()

        # Replace each change_param with its center/surround variants in-place
        for param in self._change_params:
            idx = prf_params.index(param)
            prf_params[idx : idx + 1] = [f"{param}_center", f"{param}_surround"]

        param_names = prf_params

        for key, model in self.models.items():
            if key != "prf_model" and model is not None:
                param_names.extend(model.parameter_names)

        return list(dict.fromkeys(param_names))

    def _predict_single_response(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        suffix: str,
        dtype: str,
    ) -> Tensor:
        """Run one PRF response (stimulus encoding + optional impulse convolution)."""
        params_single = parameters.copy()
        for param in self._change_params:
            params_single[param] = parameters[f"{param}_{suffix}"]

        prf_model = cast("BasePopulationResponse", self.models["prf_model"])
        response = prf_model(stimulus, params_single, dtype=dtype)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        return response

    def predict_responses(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the each of the two responses before applying betas.

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Stacked predictions of shape (num_units, 2, num_frames).

        """
        dtype = get_dtype(dtype)
        p1 = self._predict_single_response(stimulus, parameters, "center", dtype)
        p2 = self._predict_single_response(stimulus, parameters, "surround", dtype)

        return ops.stack([p1, p2], axis=1)

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the composite model response (considering Center and Surround responses).

        Applies the temporal model to the stacked responses from predict_responses.
        When scaling_model=None, returns a simple subtraction (response_center - response_surround)

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

        stacked = self.predict_responses(stimulus, parameters, dtype=dtype)

        if self.models["scaling_model"] is not None:
            temporal_model = cast("BaseScaling", self.models["scaling_model"])
            response = temporal_model(stacked, parameters, dtype=dtype)
        else:
            response = stacked[:, 0] - stacked[:, 1]

        if self.models["regressors_model"] is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", self.models["regressors_model"])
            response = response + regressors_model(regressors, parameters, dtype=dtype)

        return response


class DivNormPRFModel(BaseCanonical[PRFStimulus]):
    r"""
    Divisive normalization population receptive field model.

    Receives two independent pRF responses (activation and normalization) and combines them
    via the divisive normalization formula.
    Parameters that should be shared between both responses (e.g. the pRF centre
    ``mu_x``, ``mu_y``) are listed in ``shared_params`` and appear once in
    :attr:`parameter_names` without a suffix. All remaining pRF parameters are suffixed
    with ``_activation`` or ``_normalization`` respectively.

    Parameters
    ----------
    activation_prf_model : BaseResponse
        pRF response model (activation).
    normalization_prf_model : BaseResponse
        pRF response model (normalization).
    shared_params : list of str, default=["mu_x", "mu_y"]
        Names of pRF parameters that are shared between the two responses.  Each name
        must appear in *both* ``activation_prf_model.parameter_names`` and
        ``normalization_prf_model.parameter_names``.
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline, optional
        A scaling model class or instance. Model classes will be instantiated during initialization.
        The default creates a :class:`~prfmodel.scaling.Baseline` instance.
    %(model_regressors)s

    Notes
    -----
    The predicted response is:

    .. math::

        p_{\text{DN}} = \frac{(a R_1 \cdot S + b)}{(c R_2 \cdot S + d)} - \frac{b}{d}

    Where `R_1` and `R_2` are the activation and normalization pRF responses, `S` is the stimulus.
    The :math:`-b/d` term ensures a zero response in the absence of a stimulus.

    """

    def __init__(  # noqa: PLR0913
        self,
        activation_prf_model: BasePopulationResponse,
        normalization_prf_model: BasePopulationResponse,
        shared_params: list[str] | None = None,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        if isinstance(regressors_model, list):
            regressors_model = RegressorsList(regressors_model)

        act_names = activation_prf_model.parameter_names
        norm_names = normalization_prf_model.parameter_names
        invalid = [p for p in shared_params if p not in act_names or p not in norm_names]

        if invalid:
            msg = (
                f"Shared parameters {invalid} not found in both "
                f"'activation_prf_model.parameter_names' {act_names} and "
                f"'normalization_prf_model.parameter_names' {norm_names}"
            )
            raise ValueError(msg)

        self.shared_params = shared_params

        super().__init__(
            activation_prf_model=activation_prf_model,
            normalization_prf_model=normalization_prf_model,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Shared parameters appear once (no suffix). Response-specific parameters are suffixed
        with ``_activation`` or ``_normalization``.

        """
        shared = set(self.shared_params)
        act_model = cast("BasePopulationResponse", self.models["activation_prf_model"])
        norm_model = cast("BasePopulationResponse", self.models["normalization_prf_model"])

        param_names: list[str] = [
            "amplitude_activation",
            "amplitude_normalization",
            "baseline_activation",
            "baseline_normalization",
        ]

        # Activation model params: shared appear as-is, non-shared get _activation suffix
        for p in act_model.parameter_names:
            if p in shared:
                param_names.append(p)
            else:
                param_names.append(f"{p}_activation")

        # Normalization model non-shared params get _normalization suffix
        param_names.extend(f"{p}_normalization" for p in norm_model.parameter_names if p not in shared)

        # Encoding, impulse, and scaling model params
        for key, model in self.models.items():
            if key not in ("activation_prf_model", "normalization_prf_model") and model is not None:
                param_names.extend(model.parameter_names)

        return list(dict.fromkeys(param_names))

    def _predict_single_response(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        suffix: str,
        dtype: str,
    ) -> Tensor:
        """Predict a single encoded population receptive field model response.

        This function is both used to predict the response of the activation and normalization pRF model.

        """
        prf_model = cast("BasePopulationResponse", self.models[f"{suffix}_prf_model"])

        shared_params = set(self.shared_params)

        # Build a parameter slice for this pRF model: copy all params, then
        # overwrite non-shared params from the suffixed columns.
        params_single = parameters.copy()

        for p in prf_model.parameter_names:
            if p not in shared_params:
                params_single[p] = parameters[f"{p}_{suffix}"]

        response = prf_model(stimulus, params_single, dtype=dtype)

        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])

        return encoding_model(stimulus, response, parameters, dtype=dtype)

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the divisive normalization model response to a stimulus.

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

        regressors_model = self.models["regressors_model"]

        if regressors_model is None and regressors is not None:
            msg = "'regressors' was provided but 'regressors_model' is not configured on this model"
            raise ValueError(msg)

        if regressors_model is not None and regressors is None:
            msg = "'regressors' must be provided when 'regressors_model' is configured on this model"
            raise ValueError(msg)

        a = convert_parameters_to_tensor(parameters[["amplitude_activation"]], dtype=dtype)
        c = convert_parameters_to_tensor(parameters[["amplitude_normalization"]], dtype=dtype)

        b = convert_parameters_to_tensor(parameters[["baseline_activation"]], dtype=dtype)
        d = convert_parameters_to_tensor(parameters[["baseline_normalization"]], dtype=dtype)

        response_activation = a * self._predict_single_response(stimulus, parameters, "activation", dtype) + b
        response_normalization = c * self._predict_single_response(stimulus, parameters, "normalization", dtype) + d

        response = response_activation / response_normalization - b / d

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["scaling_model"] is not None:
            scaling_model = cast("BaseScaling", self.models["scaling_model"])
            response = scaling_model(response, parameters, dtype=dtype)

        if regressors_model is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", regressors_model)
            response = response + regressors_model(regressors, parameters, dtype=dtype)

        return response
