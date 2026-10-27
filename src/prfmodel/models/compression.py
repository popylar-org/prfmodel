"""Generic compressive models.

This module contains classes for (de-) compressing stimulus-encoded model responses.

Compressive models are intented to be used as encoding submodels within canonical models, e.g.,
:class:`~prfmodel.models.prf.canonical.CanonicalPRFModel`.

Notes
-----
Classes in this module are generic, that is, they must define the user-facing and tensor-holding stimulus types
that :meth:`call` takes as input (see also :mod:`~prfmodel.models.base`).

"""

from typing import cast
from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.models.base import S
from prfmodel.models.base import T
from prfmodel.protocols import CompositeModelProtocol
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame


class CompressiveEncoder(CompositeModelProtocol, BaseStimulusEncoder[S, T]):
    r"""
    Compressive encoding model.

    Compresses an encoded stimulus response.
    The model has one parameter: `n` (compression exponent).

    Parameters
    ----------
    encoding_model : BaseEncoder
        A encoding model instance.
    min_response : float, default=1e-10
        Minimum encoded response (:math:`\epsilon`). A small value ensures numerical stability of gradients when
        :math:`n < 1`.

    Notes
    -----
    Compressive encoding with exponent :math:`n` is done according to the equation [1]_:

    .. math::

        p(x) = \text{max}(f(x), \epsilon)^n

    The equation in [1]_ also has a gain factor. The model has no such parameter because in a canonical model the
    compressed response is scaled after the impulse stage (e.g., by the ``amplitude`` of
    :class:`~prfmodel.scaling.BaselineAmplitude`), and a gain would trade off exactly against that scale.

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
    ...     "n": [0.4, 0.5, 0.9],
    ... })
    >>> # Predict model response
    >>> resp = model(stimulus, prf_response, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 170)

    """

    _additional_parameter_names = ("n",)

    def __init__(self, encoding_model: BaseStimulusEncoder, min_response: float = 1e-10):
        self.models = {"encoding_model": encoding_model}
        self.min_response = min_response

    @doc
    def call(self, stimulus: T, response: Tensor, parameters: TensorFrame) -> Tensor:
        """Compress and encode a model response with a stimulus.

        Encodes the model response, then compresses the encoded response.

        Parameters
        ----------
        %(stimulus_tensors)s
        response : Tensor
            Model response.
        %(parameters_tensors)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The compressed and stimulus encoded model response with shape `(num_units, ...)`.
            The number of units is the number of rows in `parameters`. The number and size of other axes depends on
            the stimulus and the response.

        """
        n = parameters[["n"]]
        encoding_model = cast("BaseStimulusEncoder", self.models["encoding_model"])
        response = encoding_model.call(stimulus, response, parameters)
        return ops.power(ops.maximum(response, self.min_response), n)
