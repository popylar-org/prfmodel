from typing import cast
import pandas as pd
from keras import ops
from prfmodel.models.base import BaseImpulse
from prfmodel.models.base import BasePRFModel
from prfmodel.models.base import BaseTemporal
from prfmodel.models.encoding import encode_prf_response
from prfmodel.models.gaussian import Gaussian2DResponse
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.impulse import convolve_prf_impulse_response
from prfmodel.models.temporal import DoGAmplitude
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype


class DoG2DPRFModel(BasePRFModel):
    """
    Two-dimensional difference of Gaussians population receptive field model.

    Runs two Gaussian pipelines (sigma1 and sigma2) through encode and convolve
    independently, then combines them as a linear model:
    y(t) = p1(t) * beta_1 + p2(t) * beta_2 + baseline

    Parameters
    ----------
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DoGAmplitude
        A temporal model class or instance.

    """

    def __init__(
        self,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DoGAmplitude,
    ):
        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

        super().__init__(
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )
        self._response_model = Gaussian2DResponse()

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters used by the model."""
        param_names = ["mu_y", "mu_x", "sigma1", "sigma2"]

        for model in self.models.values():
            if model is not None:
                param_names.extend(model.parameter_names)

        return list(dict.fromkeys(param_names))

    def _predict_single_response(
        self, stimulus: Stimulus, parameters: pd.DataFrame, sigma_col: str, dtype: str,
    ) -> Tensor:
        """Run one Gaussian through encode + convolve."""
        params_single = parameters[["mu_y", "mu_x"]].copy()
        params_single["sigma"] = parameters[sigma_col]

        response = self._response_model(stimulus, params_single, dtype=dtype)
        design = ops.convert_to_tensor(stimulus.design, dtype=dtype)
        response = encode_prf_response(response, design, dtype=dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        return response

    def predict_responses(
        self, stimulus: Stimulus, parameters: pd.DataFrame, dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the two pipeline responses before applying betas.

        Returns
        -------
        Tensor
            Stacked predictions of shape (num_voxels, 2, num_frames).

        """
        dtype = get_dtype(dtype)
        p1 = self._predict_single_response(stimulus, parameters, "sigma1", dtype)
        p2 = self._predict_single_response(stimulus, parameters, "sigma2", dtype)

        return ops.stack([p1, p2], axis=1)

    def __call__(
        self, stimulus: Stimulus, parameters: pd.DataFrame, dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the DoG composite model response.

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

        # NOTE: When `temporal_model=None`, we return a simple subtraction (resp1 - resp2)
        # to remain useful without a temporal model. Not sure if this makes sense...
        return stacked[:, 0] - stacked[:, 1]