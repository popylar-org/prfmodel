"""Convolution functions."""

from keras import ops
from prfmodel.models.base import BatchDimensionError
from prfmodel.typing import Tensor


def _pad_response(response: Tensor, pad_len: int) -> Tensor:
    padding = ops.tile(response[:, :1], (1, pad_len))
    return ops.concatenate([padding, response], axis=1)


def _prepare_prf_impulse_response(prf_response: Tensor, impulse_response: Tensor) -> tuple[Tensor, Tensor]:
    # Flip impulse response on time axis
    impulse_response_flipped = ops.flip(impulse_response, axis=1)
    # We pad the pRF response signal on the left side by repeating the first response value
    # This ensures that, during the convolution, the impulse response starts at every frame of the pRF response
    # and the output shape is the same as the pRF response shape
    pad_len = impulse_response.shape[1] - 1
    prf_response_padded = _pad_response(prf_response, pad_len)

    # Transpose to meet shape requirements of depthwise convolution
    prf_response_transposed = ops.expand_dims(  # shape (1, num_frames, num_batches)
        ops.transpose(prf_response_padded),
        0,
    )
    # Transpose and flip impulse response on batch axis
    impulse_response_transposed = ops.flip(  # shape (num_frames, num_batches, 1)
        ops.expand_dims(ops.transpose(impulse_response_flipped), -1),
        axis=1,
    )

    return prf_response_transposed, impulse_response_transposed


def convolve_prf_impulse_response(prf_response: Tensor, impulse_response: Tensor) -> Tensor:
    """
    Convolve the encoded response from a population receptive field model with an impulse response.

    Both responses must have the same number of batches but can have different numbers of frames.

    Parameters
    ----------
    prf_response : Tensor
        Encoded population receptive field model response. Must have shape (num_batches, num_response_frames).
    impulse_response : Tensor
        Impulse response. Must have shape (num_batches, num_impulse_frames).

    Returns
    -------
    Tensor
        Convolved response with shape (num_batches, num_response_frames).

    Notes
    -----
    Before convolving both responses, the `prf_response` is padded on the left side in the
    `num_frames` dimension by repeating the first value of each batch. This ensures that the output of the convolution
    has the same shape as `prf_response` and the `impulse_response` starts at every frame of the `prf_response`.

    Raises
    ------
    BatchDimensionError
        If `prf_response` and `impulse_response` have batch (first) dimensions with different sizes.

    """
    prf_response = ops.convert_to_tensor(prf_response)
    impulse_response = ops.convert_to_tensor(impulse_response)

    if prf_response.shape[0] != impulse_response.shape[0]:
        raise BatchDimensionError(
            arg_names=("prf_response", "impulse_response"),
            arg_shapes=(prf_response.shape, impulse_response.shape),
        )

    # Flip, pad, and transpose responses
    prf_response_transposed, impulse_response_transposed = _prepare_prf_impulse_response(
        prf_response,
        impulse_response,
    )

    # We perform 1D depthwise convolution
    response_conv = ops.depthwise_conv(
        prf_response_transposed,
        impulse_response_transposed,  # Flip along time axis for convolution
        padding="valid",
    )

    # Transpose back and remove first dummy dimension: (num_batches, num_frames)
    return ops.transpose(response_conv[0, :, :])
