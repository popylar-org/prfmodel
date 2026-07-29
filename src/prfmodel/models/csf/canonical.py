"""Canonical contrast sensitivity function (CSF) models.

This module contains models that combine multiple exchangeable submodels in a way that is considered "canonical".

"""

from typing import cast
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BasePopulationResponse
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _normalize_regressors_model
from prfmodel.regressors.base import _validate_regressors_argument
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import CSFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype


class CanonicalCSFModel(BaseCanonical[CSFStimulus]):
    """
    Simple composite contrast sensitivity function model.

    This is a generic class that combines a contrast sensitivity function response, impulse, and
    temporal model response.

    Parameters
    ----------
    %(model_csf)s
    %(model_impulse)s
    %(model_scaling)s
    %(model_regressors)s

    Notes
    -----
    The canonical model follows the following steps:

    1. The CSF response model predicts the temporal response from per-frame spatial frequency and contrast.
    3. The impulse model generates an impulse response.
    4. The CSF response is convolved with the impulse response.
    5. The scaling model modifies the convolved response.
    6. The regressors model (optional) adds a linear combination of fixed regressors to the scaled response.

    """

    def __init__(
        self,
        csf_model: BasePopulationResponse,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = BaselineAmplitude,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        regressors_model = _normalize_regressors_model(regressors_model)

        super().__init__(
            csf_model=csf_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @doc
    def __call__(
        self,
        stimulus: CSFStimulus,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a simple contrast sensitivity function model response to a stimulus.

        Parameters
        ----------
        %(stimulus_csf)s
        %(parameters)s
        %(regressors_canonical)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        _validate_regressors_argument(self.models["regressors_model"], regressors)

        csf_model = cast("BasePopulationResponse", self.models["csf_model"])
        response = csf_model(stimulus, parameters, dtype=dtype)

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
