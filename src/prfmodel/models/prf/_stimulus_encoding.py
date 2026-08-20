"""Population receptive field stimulus encoding."""

from keras import ops
from prfmodel._docstring import doc
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.stimuli import PRFStimulus
from prfmodel.stimuli import PRFStimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import get_dtype


@doc
def encode_prf_response(response: Tensor, design: Tensor, dtype: str | None = None) -> Tensor:
    """
    Encode a population receptive field model response with a stimulus design.

    Multiplies a stimulus design with a neuron population response along the
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
    ShapeMismatchError
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
        raise ShapeMismatchError("response", response.shape, "design", design.shape)  # noqa: EM101 (exception literal)

    design = ops.expand_dims(design, 0)
    response = ops.expand_dims(response, 1)
    # Do not sum over the first two dimensions: num_units, num_frames. These are axis indices rather than
    # data, so they are built with 'range': a tensor here cannot be iterated when the step is traced.
    axes = tuple(range(2, len(design.shape)))
    # tensordot is much more memory efficient that standard multiplication
    return ops.squeeze(ops.tensordot(response, design, axes=[axes, axes]), axis=(1, 2))


class PRFStimulusEncoder(BaseStimulusEncoder[PRFStimulus, PRFStimulusTensors]):
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
    def call(self, stimulus: PRFStimulusTensors, response: Tensor, parameters: TensorFrame) -> Tensor:
        """Encode a population receptive field model response with a stimulus design.

        Parameters
        ----------
        %(stimulus_prf_tensors)
        response : Tensor
            Population receptive field response.
        %(parameters_tensors)

        Returns
        -------
        %(predicted_response_2d)s

        """
        return encode_prf_response(response, stimulus.design, dtype=parameters.dtype)
