"""Gaussian connective field response models."""

import math
import numpy as np
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.models.base import BaseTuning
from prfmodel.regressors.base import BaseRegressors
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import CFStimulus
from prfmodel.stimuli import CFStimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from ._stimulus_encoding import CFStimulusEncoder
from .canonical import CanonicalCFModel


class GaussianCFTuning(BaseTuning[CFStimulus, CFStimulusTensors]):
    """
    Gaussian connective field tuning model.

    Predicts a neuron population tuning profile to a stimulus distance matrix.
    The model has two parameters: `center_index` is the index of the row in the stimulus distance matrix that is the
    center of the Gaussian; `sigma` for the width of the Gaussian.

    Notes
    -----
    Parameter `center_index` is a discrete integer parameter that is used as an index for the distrance matrix. It must
    be non-negative and below the number of rows in the distance matrix. It must not be optimized with stochastic
    gradient descent, use a grid search instead.

    This model predicts a proper Gaussian density that is normalized by its volume (so that it integrates to one).
    This differs from other Gaussian pRF model implementations. See :doc:`Important details </important_details>`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.stimuli import CFStimulus
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
    >>> model = GaussianCFTuning()
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_source_units)
    (2, 10)

    """

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `center_index`, `sigma`."""
        return ["center_index", "sigma"]

    @doc
    def check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the model is defined on.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter that must be ``> 0`` is zero or negative or when `center_index` is a decimal or ``< 0``.

        """
        center_index = parameters["center_index"].to_numpy()

        if not np.all(center_index == np.floor(center_index)):
            msg = "Parameter 'center_index' must be a whole number because it indexes a row of the distance matrix"
            raise ValueError(msg)

        if (center_index < 0).any():
            msg = "Parameter 'center_index' must be non-negative because it indexes a row of the distance matrix"
            raise ValueError(msg)

        super().check_parameter_values(parameters)

    @doc
    def call(self, stimulus: CFStimulusTensors, parameters: TensorFrame) -> Tensor:
        """
        Predict the tuning profile for a stimulus with a distance matrix.

        Parameters
        ----------
        %(stimulus_cf_tensors)s
        %(parameters_tensors)s

        Returns
        -------
        Tensor
            Model predictions of shape `(num_units, num_rows)` and dtype `dtype`.
            `num_units` is the number of rows in `parameters` and `num_rows` is the number of rows in the stimulus
            distance matrix.

        """
        # The row is selected with 'ops.take' rather than by indexing the NumPy distance matrix, so that
        # the selection also works when 'center_index' arrives as a tensor, which it does whenever this
        # runs inside a compiled function. Gathering by index still admits no gradient, so 'center_index'
        # remains estimable by grid search only.
        center_index = ops.cast(ops.round(parameters[["center_index"]]), "int32")
        sigma = parameters[["sigma"]]
        distance_matrix = ops.take(stimulus.distance_matrix, ops.reshape(center_index, (-1,)), axis=0)

        sigma_squared = ops.square(sigma)

        # Gaussian response
        resp = ops.square(distance_matrix)
        resp /= 2.0 * sigma_squared

        # Divide by volume to normalize (only two dimensions, so exponent cancels out)
        volume = 2.0 * math.pi * sigma_squared

        return ops.exp(-resp) / volume


@doc
class GaussianCFModel(CanonicalCFModel):
    """
    Gaussian connective field model.

    This is a generic class that combines a Gaussian connective field and scaling model response.

    Parameters
    ----------
    %(model_encoding_cf)s
    %(model_scaling)s
    %(model_regressors)s

    Notes
    -----
    %(gaussian_norm_note)s

    The canonical model follows the following steps [1]_:

    1. The Gaussian connective field tuning model makes a prediction for the stimulus distance matrix.
    2. The encoding model encodes the connective field tuning profile with the source response.
    3. The scaling model modifies the encoded response.
    4. The regressors model (optional) adds a linear combination of fixed regressors to the scaled response.

    Using the default scaling model, the following columns are expected in the
    :class:`pandas.DataFrame` passed as the ``parameters`` argument to :meth:`__call__`:

    .. list-table::
       :header-rows: 1
       :widths: 20 12 53

       * - Parameter
         - Model
         - Description
       * - ``center_index``
         - CF
         - Row index of the center unit in the source distance matrix.
       * - ``sigma``
         - CF
         - Standard deviation of the Gaussian.
       * - ``baseline``
         - Scaling
         - Additive constant.
       * - ``amplitude``
         - Scaling
         - Multiplicative scale factor.

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
    >>> from prfmodel.stimuli import CFStimulus
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
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = CFStimulusEncoder,
        scaling_model: BaseScaling | type[BaseScaling] | None = BaselineAmplitude,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        super().__init__(
            cf_model=GaussianCFTuning(),
            encoding_model=encoding_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
        )
