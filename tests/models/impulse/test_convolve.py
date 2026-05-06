"""Tests for convolution functions."""

import numpy as np
import pytest
from prfmodel.models.base import BatchDimensionError
from prfmodel.models.impulse.convolve import _pad_response
from prfmodel.models.impulse.convolve import _prepare_prf_impulse_response
from prfmodel.models.impulse.convolve import convolve_prf_impulse_response


def test_pad_response():
    """Test that _pad_response returns response with correct shape and correct values."""
    response = np.expand_dims(np.arange(5), 0)  # shape (1, 5)
    pad_len = 2

    response_padded = np.asarray(_pad_response(response, pad_len))

    assert response_padded.shape == (1, response.shape[1] + pad_len)
    assert np.all(response_padded[:, :pad_len] == response[:, 0])
    assert np.all(response_padded[:, pad_len:] == response)


def test_prepare_prf_impulse_response():
    """Test that _prepare_prf_impulse_response returns responses with correct shapes and correct values."""
    prf_response = np.expand_dims(np.arange(20), 0)
    pad_len = 2
    impulse_response = np.expand_dims(np.arange(pad_len + 1), 0)

    prf_response_transposed, impulse_response_transposed = _prepare_prf_impulse_response(
        prf_response,
        impulse_response,
    )
    prf_response_transposed = np.asarray(prf_response_transposed)
    impulse_response_transposed = np.asarray(impulse_response_transposed)

    assert prf_response_transposed.shape == (1, prf_response.shape[1] + pad_len, 1)  # shape (1, 22, 1)
    assert np.all(prf_response_transposed[:, pad_len:, 0] == prf_response)
    assert np.all(prf_response_transposed[:, :pad_len, 0] == prf_response[:, :1])
    assert impulse_response_transposed.shape == (*np.transpose(impulse_response).shape, 1)  # shape (3, 1, 1)
    assert np.all(impulse_response_transposed[:, 0, 0] == np.flip(impulse_response[0, :]))


def test_convolve_prf_impulse_response():
    """Test that convolve_prf_impulse_response returns response with correct shape."""
    num_batches = 3
    num_prf_frames = 10
    num_irf_frames = 3

    prf_response = np.ones((num_batches, num_prf_frames))
    irf_response = np.ones((num_batches, num_irf_frames))

    resp_conv = convolve_prf_impulse_response(prf_response, irf_response)

    assert resp_conv.shape == (num_batches, num_prf_frames)


def test_convolve_prf_impulse_response_batch_dimension_error():
    """Test that convolve_prf_impulse_response raises error when batch dimension does not match."""
    with pytest.raises(BatchDimensionError):
        _ = convolve_prf_impulse_response(np.ones((20, 10)), np.ones((10, 3)))
