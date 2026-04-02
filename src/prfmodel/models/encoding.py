"""Stimulus encoding classes."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.models.base import BaseEncoder
from prfmodel.models.base import S
from prfmodel.stimuli.cf import CFStimulus
from prfmodel.stimuli.prf import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype


class ResponseDesignShapeError(Exception):
    """
    Exception raised when the shapes of the model response and stimulus design do not match.

    Both must have the same shape after the first dimension.

    Parameters
    ----------
    response_shape : tuple of int
        Shape of the model response array.
    design_shape : tuple of int
        Shape of the design array.
    """

    def __init__(self, response_shape: tuple[int, ...], design_shape: tuple[int, ...]):
        super().__init__(f"Shapes of 'response' {response_shape} and 'design' {design_shape} do not match")


@doc
def encode_prf_response(response: Tensor, design: Tensor, dtype: str | None = None) -> Tensor:
    """
    Encode a population receptive field model response with a stimulus design.

    Multiplies a stimulus design with a model response along the
    stimulus dimensions and sums over them.

    Parameters
    ----------
    response : Tensor
        The population receptive field model response. The first dimension corresponds to the number of batches.
        Additional dimensions correspond to the stimulus dimensions.
    design : Tensor
        The stimulus design containing the stimulus value in one or more dimensions over different time frames.
        The first axis is assumed to be time frames. Additional axes represent stimulus dimensions.
    %(dtype)s

    Returns
    -------
    :data:`prfmodel.typing.Tensor`
        The stimulus encoded model response with shape (num_units, num_frames) and dtype `dtype`.

    Raises
    ------
    ResponseDesignShapeError
        If the shape of the model response and the stimulus design do not match.

    Examples
    --------
    Encode a 1D model response:

    >>> import numpy as np
    >>> num_units = 3
    >>> num_frames = 10
    >>> height = 5
    >>> # Create a dummy stimulus design
    >>> design = np.ones((num_frames, height))
    >>> # Create a dummy model response that varies with the height of a stimulus grid
    >>> resp = np.ones((num_units, height)) * np.expand_dims(np.sin(np.arange(height)), 0)
    >>> print(resp.shape) # (num_units, height)
    (3, 5)
    >>> resp_encoded = encode_prf_response(resp, design)
    >>> print(resp_encoded.shape)  # (num_units, num_frames)
    (3, 10)

    Encode a 2D model response:

    >>> # Add width dimension
    >>> width = 4
    >>> # Create a dummy stimulus design
    >>> design = np.ones((num_frames, height, width))
    >>> # Create a dummy model response that varies with the width of a stimulus grid
    >>> resp = np.ones((num_units, height, width)) * np.expand_dims(np.sin(np.arange(width)), (0, 1))
    >>> print(resp.shape) # (num_units, height, width)
    (3, 5, 4)
    >>> resp_encoded = encode_prf_response(resp, design)
    >>> print(resp_encoded.shape)  # (num_units, num_frames)
    (3, 10)

    """
    dtype = get_dtype(dtype)
    response = ops.convert_to_tensor(response, dtype)
    design = ops.convert_to_tensor(design, dtype)

    if response.shape[1:] != design.shape[1:]:
        raise ResponseDesignShapeError(response.shape, design.shape)

    design = ops.expand_dims(design, 0)
    response = ops.expand_dims(response, 1)
    # Do not sum over the first two dimensions: num_units, num_frames
    axes = tuple(ops.arange(2, len(design.shape)))
    # tensordot is much more memory efficient that standard multiplication
    return ops.squeeze(ops.tensordot(response, design, axes=[axes, axes]), axis=(1, 2))


class PRFStimulusEncoder(BaseEncoder[PRFStimulus]):
    """
    Encoding model for population receptive field stimuli.

    Multiplies a stimulus design with a population receptive field model response along the
    stimulus dimensions and sums over them.

    See Also
    --------
    encode_prf_response

    """

    @property
    def parameter_names(self) -> list:
        """Does not have any parameters. Returns an empty list."""
        return []

    @doc
    def __call__(
        self,
        stimulus: PRFStimulus,
        response: Tensor,
        parameters: pd.DataFrame,  # noqa: ARG002 (unused method argument)
        dtype: str | None = None,
    ) -> Tensor:
        """Encode a population receptive field model response with a stimulus design.

        Parameters
        ----------
        %(stimulus_prf)s
        response : Tensor
            Population receptive field response.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        design = ops.convert_to_tensor(stimulus.design, dtype=dtype)
        return encode_prf_response(response, design, dtype=dtype)


class CFStimulusEncoder(BaseEncoder[CFStimulus]):
    """
    Encoding model for connective field stimuli.

    Multiplies a source response with a connective model response and sums over the vertices in the source response.

    """

    @property
    def parameter_names(self) -> list:
        """Does not have any parameters. Returns an empty list."""
        return []

    @doc
    def __call__(
        self,
        stimulus: CFStimulus,
        response: Tensor,
        parameters: pd.DataFrame,  # noqa: ARG002 (unused method argument)
        dtype: str | None = None,
    ) -> Tensor:
        """Encode a Connective field model response with a source response.

        Parameters
        ----------
        %(stimulus_cf)s
        response : Tensor
            Connective field response.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        response = ops.convert_to_tensor(response, dtype=dtype)
        source_response = ops.convert_to_tensor(stimulus.source_response, dtype=dtype)

        if response.shape[1] != source_response.shape[0]:
            msg = (
                f"Second dimension of connective field response {response.shape[1]} does not match first dimension "
                f"of source response {source_response.shape[0]}"
            )
            raise ValueError(msg)

        return ops.tensordot(response, source_response, axes=[[1], [0]])


class CompressiveEncoder(BaseEncoder[S]):
    r"""
    Compressive encoding model.

    Amplifies and compresses an encoded stimulus response.
    The model has two parameters: `gain` (amplification amplitude) and `n` (compression exponent).

    Parameters
    ----------
    encoding_model : BasePRFEncoder
        A encoding model instance.
    min_response : float, default=1e-10
        Minimum encoded response (:math:`\epsilon`). A small value ensures numerical stability of gradients when
        :math:`n < 1`.

    Notes
    -----
    Compressive encoding with `gain` :math:`g` and :math:`n` is done according to the equation:

    .. math::

        p(x) = g \times \text{max}(f(x), \epsilon)^n

    References
    ----------
    .. [1] Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual
        cortex. *Journal of Neurophysiology*, 110(2), 481-494. https://doi.org/10.1152/jn.00105.2013

    Examples
    --------
    Predict a model response for multiple units.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models.encoding import CompressiveEncoder, PRFStimulusEncoder
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> print(stimulus)
    PRFStimulus(design=array[200, 101, 101], grid=array[101, 101, 2], dimension_labels=['y', 'x'])
    >>> # Create dummy response as input for encdoder
    >>> prf_response = np.ones((3, 101, 101))  # Must have same number of frames as stimulus
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
    (3, 200)

    """

    def __init__(self, encoding_model: BaseEncoder, min_response: float = 1e-10):
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
        %(stimulus)s
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
