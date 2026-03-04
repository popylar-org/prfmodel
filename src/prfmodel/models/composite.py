"""Composite models."""

from typing import cast
import pandas as pd
from keras import ops
from prfmodel.stimuli.cf import CFStimulus
from prfmodel.stimuli.prf import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .base import BaseCFResponse
from .base import BaseComposite
from .base import BaseImpulse
from .base import BasePRFResponse
from .base import BaseTemporal
from .encoding import encode_prf_response
from .impulse import DerivativeTwoGammaImpulse
from .impulse import convolve_prf_impulse_response
from .temporal import BaselineAmplitude
from .temporal import DoGAmplitude


class SimplePRFModel(BaseComposite):
    """
    Simple composite population receptive field model.

    This is a generic class that combines a population receptive field, impulse, and temporal response.

    Parameters
    ----------
    prf_model : BasePRFResponse
        A population receptive field response model instance.
    impulse_model : BaseImpulse or type or None, default=ShiftedDerivativeGammaImpulse, optional
        An impulse response model class or instance. Reponse model classes will be instantiated during object
        initialization. The default creates a `ShiftedDerivativeGammaImpulse` instance with default values.
        values.
    temporal_model : BaseTemporal or type or None, default=BaselineAmplitude, optional
        A temporal model class or instance. Temporal model instances will be instantiated during initialization.
        The default creates a `BaselineAmplitude` instance.

    Notes
    -----
    The simple composite model follows five steps:

    1. The population receptive field response model makes a prediction for the stimulus grid.
    2. The response is encoded with the stimulus design.
    3. The impulse response model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The temporal model modifies the convolved response.

    """

    def __init__(
        self,
        prf_model: BasePRFResponse,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

        super().__init__(
            prf_model=prf_model,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

    def __call__(
        self,
        # We can safely override the type check here
        stimulus: PRFStimulus,  # type: ignore[override]
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a simple population receptive field model response to a stimulus.

        Parameters
        ----------
        stimulus : PRFStimulus
            Population receptive field stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different (sub-) model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape (num_voxels, num_frames) and dtype `dtype`. The number of voxels is the
            number of rows in `parameters`. The number of frames is the number of frames in the stimulus design.

        """
        dtype = get_dtype(dtype)
        prf_model = cast("BasePRFResponse", self.models["prf_model"])
        response = prf_model(stimulus, parameters, dtype=dtype)
        design = ops.convert_to_tensor(stimulus.design, dtype=dtype)
        response = encode_prf_response(response, design, dtype=dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            response = temporal_model(response, parameters, dtype=dtype)

        return response


class CenterSurroundPRFModel(BaseComposite):
    """
    Center-surround composite population receptive field model.

    This is a generic class that runs two PRF responses through stimulus encoding and impulse response convolution
    independently, then combines them via a temporal model:
    y(t) = p1(t) * amplitude_center + p2(t) * amplitude_sorround + baseline

    The two responses differ in the values of ``change_params``. Each parameter
    ``p`` in that list is split into ``{p}_center`` and ``{p}_sorround``.

    Parameters
    ----------
    prf_model : BasePRFResponse
        A population receptive field response model instance.
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DoGAmplitude
        A temporal model class or instance.
    change_params : list[str], default=["sigma"]
        Names of the parameters that differ between the center and surround responses.
        All entries must be present in ``prf_model.parameter_names``.

    Notes
    -----
    The center-surround composite model follows these steps:

    1. Two PRF response predictions are computed, one using ``{p}_center`` and one
       using ``{p}_sorround`` for each ``p`` in ``change_params``.
    2. Each response is encoded with the stimulus design.
    3. Each encoded response is optionally convolved with an impulse response.
    4. The two responses are stacked and combined by the temporal model.

    """

    def __init__(
        self,
        prf_model: BasePRFResponse,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DoGAmplitude,
        change_params: list[str] | None = None,
    ):
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
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters used by the model."""
        prf_model = cast("BasePRFResponse", self.models["prf_model"])
        prf_params = prf_model.parameter_names.copy()

        # Replace each change_param with its center/sorround variants in-place
        for param in self._change_params:
            idx = prf_params.index(param)
            prf_params[idx : idx + 1] = [f"{param}_center", f"{param}_sorround"]

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

        prf_model = cast("BasePRFResponse", self.models["prf_model"])
        response = prf_model(stimulus, params_single, dtype=dtype)
        design = ops.convert_to_tensor(stimulus.design, dtype=dtype)
        response = encode_prf_response(response, design, dtype=dtype)

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
        Tensor
            Stacked predictions of shape (num_voxels, 2, num_frames).

        """
        dtype = get_dtype(dtype)
        p1 = self._predict_single_response(stimulus, parameters, "center", dtype)
        p2 = self._predict_single_response(stimulus, parameters, "sorround", dtype)

        return ops.stack([p1, p2], axis=1)

    def __call__(
        self,
        stimulus: PRFStimulus,  # type: ignore[override]
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the composite model response (considering Center and Sorround responses).

        Applies the temporal model to the stacked responses from predict_responses.
        When temporal_model=None, returns a simple subtraction (response_center - response_sorround)

        Parameters
        ----------
        stimulus : PRFStimulus
            Population receptive field stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different (sub-) model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the prediction result. If ``None`` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape (num_voxels, num_frames).

        """
        dtype = get_dtype(dtype)
        stacked = self.predict_responses(stimulus, parameters, dtype=dtype)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            return temporal_model(stacked, parameters, dtype=dtype)

        # When temporal_model=None, return a simple subtraction (resp1 - resp2)
        return stacked[:, 0] - stacked[:, 1]


class SimpleCFModel(BaseComposite):
    """
    Simple composite connective field model.

    This is a generic class that combines a connective field and temporal response.

    Parameters
    ----------
    cf_model : BaseCFResponse
        A connective field response model instance.
    temporal_model : BaseTemporal or type or None, default=BaselineAmplitude, optional
        A temporal model class or instance. Temporal model instances will be instantiated during initialization.
        The default creates a `BaselineAmplitude` instance.

    Notes
    -----
    The simple composite model follows three steps:

    1. The connective field response model makes a prediction for the stimulus distance matrix.
    2. The connective field response is encoded with the source response.
    3. The temporal model modifies the encoded response.

    """

    def __init__(
        self,
        cf_model: BaseCFResponse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

        super().__init__(
            cf_model=cf_model,
            temporal_model=temporal_model,
        )

    def __call__(
        self,
        stimulus: CFStimulus,  # type: ignore[override]
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a simple connective field model response to a stimulus.

        Parameters
        ----------
        stimulus : CFStimulus
            Connective field stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different (sub-) model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape (num_voxels, num_frames) and dtype `dtype`. The number of voxels is the
            number of rows in `parameters`. The number of frames is the number of frames in the stimulus source
            response.

        """
        dtype = get_dtype(dtype)
        cf_model = cast("BaseCFResponse", self.models["cf_model"])
        response = cf_model(stimulus, parameters, dtype=dtype)
        response = ops.expand_dims(response, -1)
        source_response = ops.convert_to_tensor(stimulus.source_response, dtype=dtype)
        source_response = ops.expand_dims(source_response, 0)
        response *= source_response
        response = ops.sum(response, axis=1)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            response = temporal_model(response, parameters, dtype=dtype)

        return response
