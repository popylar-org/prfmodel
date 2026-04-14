"""Canonical populationr receptive field (pRF) models."""

from typing import cast
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BaseEncoder
from prfmodel.models.base import BaseResponse
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling import DoGAmplitude
from prfmodel.scaling.base import BaseTemporal
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from ._stimulus_encoding import PRFStimulusEncoder


class CanonicalPRFModel(BaseCanonical[PRFStimulus]):
    """
    Canonical population receptive field (pRF) model.

    This class combines a pRF response, impulse, and scaling model.

    Parameters
    ----------
    %(model_prf)s
    %(model_encoding)s
    %(model_impulse)s
    %(model_temporal)s

    Notes
    -----
    The simple composite model follows five steps:

    1. The population receptive field response model makes a prediction for the stimulus grid.
    2. The encoding model encodes the response with the stimulus design.
    3. The impulse response model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The temporal model modifies the convolved response.

    """

    @doc
    def __init__(
        self,
        prf_model: BaseResponse,
        encoding_model: BaseEncoder | type[BaseEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

        super().__init__(
            prf_model=prf_model,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a simple population receptive field model response to a stimulus.

        Parameters
        ----------
        %(stimulus_prf)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        prf_model = cast("BaseResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters, dtype=dtype)
        encoding_model = cast("BaseEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            response = temporal_model(response, parameters, dtype=dtype)

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
    %(model_encoding)s
    %(model_impulse)s
    %(model_temporal)s
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
    4. The two responses are stacked and combined by the temporal model.

    """

    def __init__(
        self,
        prf_model: BaseResponse,
        encoding_model: BaseEncoder | type[BaseEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DoGAmplitude,
        change_params: list[str] | None = None,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

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
            temporal_model=temporal_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters used by the model."""
        prf_model = cast("BaseResponse", self.models["prf_model"])
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

        prf_model = cast("BaseResponse", self.models["prf_model"])
        response = prf_model(stimulus, params_single, dtype=dtype)
        encoding_model = cast("BaseEncoder", self.models["encoding_model"])
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
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the composite model response (considering Center and Surround responses).

        Applies the temporal model to the stacked responses from predict_responses.
        When temporal_model=None, returns a simple subtraction (response_center - response_surround)

        Parameters
        ----------
        %(stimulus_prf)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        stacked = self.predict_responses(stimulus, parameters, dtype=dtype)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            return temporal_model(stacked, parameters, dtype=dtype)

        # When temporal_model=None, return a simple subtraction (resp1 - resp2)
        return stacked[:, 0] - stacked[:, 1]
