"""Generic compressive models.

This module contains classes for (de-) compressing stimulus-encoded model responses.

Compressive models are intented to be used as encoding submodels within canonical models, e.g.,
:class:`~prfmodel.models.prf.canonical.CanonicalPRFModel`.

Notes
-----
Classes in this module are generic, that is, they can take arbitrary stimuli as input (e.g., they work both
for :class:`~prfmodel.stimuli.PRFStimulus` and :class:`~prfmodel.stimuli.CFStimulus`).

"""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.models.base import S
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype


class CompressiveEncoder(BaseStimulusEncoder[S]):
    r"""
    Compressive encoding model.

    Amplifies and compresses an encoded stimulus response.
    The model has two parameters: `gain` (amplification amplitude) and `n` (compression exponent).

    Parameters
    ----------
    encoding_model : BaseEncoder
        A encoding model instance.
    min_response : float, default=1e-10
        Minimum encoded response (:math:`\epsilon`). A small value ensures numerical stability of gradients when
        :math:`n < 1`.

    Notes
    -----
    Compressive encoding with `gain` :math:`g` and :math:`n` is done according to the equation [1]_:

    .. math::

        p(x) = g \times \text{max}(f(x), \epsilon)^n

    References
    ----------
    .. [1] Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual
        cortex. *Journal of Neurophysiology*, 110(2), 481-494. https://doi.org/10.1152/jn.00105.2013

    Examples
    --------
    Predict an encode population receptive field model response for multiple units.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models.prf import PRFStimulusEncoder
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> # Create dummy response as input for encdoder
    >>> prf_response = np.ones((3, 128, 128))  # Must have same number of frames as stimulus
    >>> model = CompressiveEncoder(
    ...     encoding_model=PRFStimulusEncoder(),
    ... )
    >>> # Define model parameters for 3 units
    >>> params = pd.DataFrame({
    ...     # Compressive parameters
    ...     "gain": [0.5, 0.1, 1.2],
    ...     "n": [0.4, 0.5, 0.9],
    ... })
    >>> # Predict model response
    >>> resp = model(stimulus, prf_response, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 170)

    """

    def __init__(self, encoding_model: BaseStimulusEncoder, min_response: float = 1e-10):
        self.encoding_model = encoding_model
        self.min_response = min_response

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `gain` and `n`."""
        return ["gain", "n"]

    @doc
    def __call__(
        self,
        stimulus: S,
        response: Tensor,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """Compress and encode a model response with a stimulus.

        Encodes the model response, then compresses and amplifies the encoded response.

        Parameters
        ----------
        %(stimulus)s The stimulus type must match the expected stimulus type of the wrapped encoding model instance
            (i.e., :attr:`~CompressiveEncoder.encoding_model`).
        response : Tensor
            Model response.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The compressed and stimulus encoded model response with shape `(num_units, ...)` dtype `dtype`.
            The number of units is the number of rows in `parameters`. The number and size of other axes depends on
            the stimulus and the response.

        """
        dtype = get_dtype(dtype=dtype)
        gain = convert_parameters_to_tensor(parameters[["gain"]], dtype)
        n = convert_parameters_to_tensor(parameters[["n"]], dtype)
        response = self.encoding_model(
            stimulus=stimulus,
            response=response,
            parameters=parameters,
            dtype=dtype,
        )
        return gain * ops.power(ops.maximum(response, self.min_response), n)
