"""Canonical connective field (CF) models.

This module contains models that combine multiple exchangeable submodels in a way that is considered "canonical".

"""

from typing import cast
from prfmodel._docstring import doc
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.models.base import BaseTuning
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _normalize_regressors_model
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import CFStimulus
from prfmodel.stimuli import CFStimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from ._stimulus_encoding import CFStimulusEncoder


@doc
class CanonicalCFModel(BaseCanonical[CFStimulus, CFStimulusTensors]):
    """
    Canonical connective field model.

    This class combines a connective field and scaling model response.

    Parameters
    ----------
    %(model_cf)s
    %(model_encoding_cf)s
    %(model_scaling)s
    %(model_regressors)s

    Notes
    -----
    The canonical model follows the following steps:

    1. The connective field tuning model makes a prediction for the stimulus distance matrix.
    2. The connective field tuning profile is encoded with the source response.
    3. The scaling model modifies the encoded response.
    4. The regressors model (optional) adds a linear combination of fixed regressors to the scaled response.

    In contrast to pRF models (e.g., :class:`~prfmodel.models.CanonicalPRFModel`), connective field models do not
    require an impulse model because it already contained in the signal of the source response.

    """

    def __init__(
        self,
        cf_model: BaseTuning,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = CFStimulusEncoder,
        scaling_model: BaseScaling | type[BaseScaling] | None = BaselineAmplitude,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if scaling_model is not None and isinstance(scaling_model, type):
            scaling_model = scaling_model()

        regressors_model = _normalize_regressors_model(regressors_model)

        super().__init__(
            cf_model=cf_model,
            encoding_model=encoding_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )

    @doc
    def call(
        self,
        stimulus: CFStimulusTensors,
        parameters: TensorFrame,
        regressors: TensorFrame | None = None,
    ) -> Tensor:
        """
        Predict a canonical connective field model response to a stimulus.

        Parameters
        ----------
        %(stimulus_cf_tensors)s
        %(parameters_tensors)s
        %(regressors_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        regressors_model = self.models["regressors_model"]

        cf_model = cast("BaseTuning", self.models["cf_model"])
        response = cf_model.call(stimulus, parameters)
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model.call(stimulus, response, parameters)

        if self.models["scaling_model"] is not None:
            temporal_model = cast("BaseScaling", self.models["scaling_model"])
            response = temporal_model.call(response, parameters)

        if regressors_model is not None and regressors is not None:
            regressors_model = cast("BaseRegressors", regressors_model)
            response = response + regressors_model.call(regressors, parameters)

        return response
