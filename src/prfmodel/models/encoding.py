"""Stimulus encoding classes."""

import pandas as pd
from keras import ops
from prfmodel.models.base import BaseEncoder
from prfmodel.stimuli.cf import CFStimulus
from prfmodel.stimuli.prf import PRFStimulus
from prfmodel.typing import Tensor
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
    dtype : str, optional
        The dtype of the prediction result. If `None` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.

    Returns
    -------
    Tensor
        The stimulus encoded model response with shape (num_batches, num_frames) and dtype `dtype`.

    Raises
    ------
    ResponseDesignShapeError
        If the shape of the model response and the stimulus design do not match.

    Examples
    --------
    Encode a 1D model response:

    >>> import numpy as np
    >>> num_batches = 3
    >>> num_frames = 10
    >>> height = 5
    >>> # Create a dummy stimulus design
    >>> design = np.ones((num_frames, height))
    >>> # Create a dummy model response that varies with the height of a stimulus grid
    >>> resp = np.ones((num_batches, height)) * np.expand_dims(np.sin(np.arange(height)), 0)
    >>> print(resp.shape) # (num_batches, height)
    (3, 5)
    >>> resp_encoded = encode_prf_response(resp, design)
    >>> print(resp_encoded.shape) (num_batches, num_frames)
    (3, 10)

    Encode a 2D model response:

    >>> # Add width dimension
    >>> width = 4
    >>> # Create a dummy stimulus design
    >>> design = np.ones((num_frames, height, width))
    >>> # Create a dummy model response that varies with the width of a stimulus grid
    >>> resp = np.ones((num_batches, height, width)) * np.expand_dims(np.sin(np.arange(width)), (0, 1))
    >>> print(resp.shape) # (num_batches, height, width)
    (3, 5, 4)
    >>> resp_encoded = encode_prf_response(resp, design)
    >>> print(resp_encoded.shape) (num_batches, num_frames)
    (3, 10)

    """
    dtype = get_dtype(dtype)
    response = ops.convert_to_tensor(response, dtype)
    design = ops.convert_to_tensor(design, dtype)

    if response.shape[1:] != design.shape[1:]:
        raise ResponseDesignShapeError(response.shape, design.shape)

    design = ops.expand_dims(design, 0)
    response = ops.expand_dims(response, 1)
    # Do not sum over the first two dimensions: num_batches, num_frames
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
        stimulus : PRFStimulus
            Population receptive field stimulus object.
        response : Tensor
            Population receptive field response.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the encoded response. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            The stimulus encoded model response with shape `(num_voxels, num_frames)` dtype `dtype`.
            The number of voxels is the number of rows in `parameters`. The number of frames is the number of time
            frames in the stimulus design.

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
        stimulus : CFStimulus
            Connective field stimulus object.
        response : Tensor
            Connective field response.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the encoded response. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            The stimulus encoded model response with shape `(num_voxels, num_frames)` dtype `dtype`.
            The number of voxels is the number of rows in `parameters`. The number of frames is the number of time
            frames in the stimulus design.

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
