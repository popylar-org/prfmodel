"""Canonical connective field (CF) models."""

from typing import cast
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BaseEncoder
from prfmodel.models.base import BaseResponse
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseTemporal
from prfmodel.stimuli import CFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .stimulus_encoding import CFStimulusEncoder


class CanonicalCFModel(BaseCanonical[CFStimulus]):
    """
    Canonical connective field model.

    This class combines a connective field response and scaling model.

    Parameters
    ----------
    %(model_cf)s
    %(model_encoding)s
    %(model_temporal)s

    Notes
    -----
    The simple composite model follows three steps:

    1. The connective field response model makes a prediction for the stimulus distance matrix.
    2. The connective field response is encoded with the source response.
    3. The temporal model modifies the encoded response.

    In contrast to pRF models (e.g., :class:`~prfmodel.models.CanonicalPRFModel`), connective field models do not
    require an impulse response model because it already contained in the signal of the source response.

    """

    def __init__(
        self,
        cf_model: BaseResponse,
        encoding_model: BaseEncoder | type[BaseEncoder] = CFStimulusEncoder,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

        super().__init__(
            cf_model=cf_model,
            encoding_model=encoding_model,
            temporal_model=temporal_model,
        )

    @doc
    def __call__(
        self,
        stimulus: CFStimulus,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a simple connective field model response to a stimulus.

        Parameters
        ----------
        %(stimulus_cf)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        cf_model = cast("BaseResponse", self.models["cf_model"])
        response = cf_model(stimulus, parameters, dtype=dtype)
        encoding_model = cast("BaseEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            response = temporal_model(response, parameters, dtype=dtype)

        return response
