"""Gaussian connective field response models."""

import math
import numpy as np
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseEncoder
from prfmodel.models.base import BaseResponse
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseTemporal
from prfmodel.stimuli import CFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .canonical import CanonicalCFModel
from .stimulus_encoding import CFStimulusEncoder


class GaussianCFResponse(BaseResponse[CFStimulus]):
    """
    Gaussian connective field response model.

    Predicts a response to a stimulus distance matrix.
    The model has two parameters: `center_index` is the index of the row in the stimulus distance matrix that is the
    center of the Gaussian; `sigma` for the width of the Gaussian.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.stimuli.cf import CFStimulus
    >>> num_source_units, num_frames = 10, 20
    >>> distances = np.abs(
    ...     np.arange(num_source_units, dtype=float)[:, None]
    ...     - np.arange(num_source_units, dtype=float)[None, :]
    ... )
    >>> source_response = np.ones((num_source_units, num_frames))
    >>> stimulus = CFStimulus(
    ...     distance_matrix=distances,
    ...     source_response=source_response
    ... )
    >>> # Define parameters for 2 target units
    >>> params = pd.DataFrame({
    ...     "center_index": [0, 5],
    ...     "sigma": [1.0, 2.0]
    ... })
    >>> model = GaussianCFResponse()
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_source_units)
    (2, 10)

    """

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `center_index`, `sigma`."""
        return ["center_index", "sigma"]

    @doc
    def __call__(self, stimulus: CFStimulus, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus with a distance matrix.

        Parameters
        ----------
        %(stimulus_cf)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        Tensor
            Model predictions of shape `(num_units, num_rows)` and dtype `dtype`.
            `num_units` is the number of rows in `parameters` and `num_rows` is the number of rows in the stimulus
            distance matrix.

        """
        dtype = get_dtype(dtype)
        # Distance matrix is numpy array so we also create a numpy array to safely index
        # The dtype is only used for indexing so it can be hardcoded
        center_index = np.asarray(parameters[["center_index"]], dtype=np.int32)[:, 0]
        sigma = convert_parameters_to_tensor(parameters[["sigma"]], dtype=dtype)
        distance_matrix = ops.convert_to_tensor(stimulus.distance_matrix[center_index], dtype=dtype)

        sigma_squared = ops.square(sigma)

        # Gaussian response
        resp = ops.square(distance_matrix)
        resp /= 2.0 * sigma_squared

        # Divide by volume to normalize (only two dimensions, so exponent cancels out)
        volume = ops.sqrt(2.0 * math.pi * sigma_squared)

        return ops.exp(-resp) / volume


class GaussianCFModel(CanonicalCFModel):
    """
    Gaussian connective field model.

    This is a generic class that combines a Gaussian connective field and temporal model response.

    Parameters
    ----------
    %(model_encoding)s
    %(model_temporal)s

    Notes
    -----
    The simple composite model follows three steps [1]_:

    1. The Gaussian connective field response model makes a prediction for the stimulus distance matrix.
    2. The encoding model encodes the connective field response with the source response.
    3. The temporal model modifies the encoded response.

    References
    ----------
    .. [1] Haak, K. V., Winawer, J., Harvey, B. M., Renken, R., Dumoulin, S. O., Wandell, B. A., &
        Cornelissen, F. W. (2013). Connective field modeling. *NeuroImage*, 66, 376-384.
        https://doi.org/10.1016/j.neuroimage.2012.10.037


    Examples
    --------
    Predict a model response for multiple units.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.models import GaussianCFModel
    >>> from prfmodel.stimuli.cf import CFStimulus
    >>> num_source_units, num_frames = 10, 20
    >>> distances = np.abs(
    ...     np.arange(num_source_units, dtype=float)[:, None]
    ...     - np.arange(num_source_units, dtype=float)[None, :]
    ... )
    >>> source_response = np.ones((num_source_units, num_frames))
    >>> stimulus = CFStimulus(
    ...     distance_matrix=distances,
    ...     source_response=source_response
    ... )
    >>> model = GaussianCFModel()
    >>> # Define parameters for 2 target units
    >>> params = pd.DataFrame({
    ...     # Gaussian parameters
    ...     "center_index": [0, 5],
    ...     "sigma": [1.0, 2.0],
    ...     # Temporal model parameters
    ...     "baseline": [0.0, 0.0],
    ...     "amplitude": [1.0, 1.0],
    ... })
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)
    (2, 20)

    """

    def __init__(
        self,
        encoding_model: BaseEncoder | type[BaseEncoder] = CFStimulusEncoder,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        super().__init__(
            cf_model=GaussianCFResponse(),
            encoding_model=encoding_model,
            temporal_model=temporal_model,
        )
