"""Tests for convolution functions."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.impulse import TwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse._convolve import _pad_response
from prfmodel.impulse._convolve import _prepare_prf_impulse_response


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
    with pytest.raises(ShapeMismatchError):
        _ = convolve_prf_impulse_response(np.ones((20, 10)), np.ones((10, 3)))


@pytest.mark.parametrize("offset", [-1.0, -3.0, -5.0])
def test_negative_impulse_offset_shifts_the_kernel_exactly(offset: float):
    """Test that a negative offset re-samples the same impulse, shifted, with zeros in front.

    Unnormalized, so that the comparison is between the sampled density values themselves. The kernel
    holds `num_frames = int(duration / resolution)` samples whatever the offset, so shifting the window
    back by `lag` frames also truncates `lag` frames off the tail.

    """
    params = pd.DataFrame(index=range(1))
    lag = int(-offset)

    baseline = np.asarray(TwoGammaImpulse(norm=None)(params))
    shifted = np.asarray(TwoGammaImpulse(offset=offset, norm=None)(params))

    assert np.all(shifted[:, :lag] == 0.0)
    np.testing.assert_allclose(shifted[:, lag:], baseline[:, :-lag], rtol=1e-6)


@pytest.mark.parametrize("offset", [-1.0, -3.0, -5.0])
def test_negative_impulse_offset_delays_the_response_by_that_many_frames(offset: float):
    """Test that a negative impulse offset is exactly a lag on the convolved response.

    `convolve_prf_impulse_response` reads `impulse_response[:, 0]` as the kernel value at lag 0, so the
    kernel carries no time origin of its own. The leading zeros from a negative `offset` are what turn it
    into a delay -- the counterpart of a positive `offset`, which advances the response by skipping the
    front of the impulse.

    """
    params = pd.DataFrame(index=range(1))
    lag = int(-offset)

    stimulus = np.zeros((1, 40))
    stimulus[0, 10] = 1.0

    baseline = np.asarray(convolve_prf_impulse_response(stimulus, TwoGammaImpulse(norm=None)(params)))
    delayed = np.asarray(convolve_prf_impulse_response(stimulus, TwoGammaImpulse(offset=offset, norm=None)(params)))

    assert int(delayed.argmax()) == int(baseline.argmax()) + lag
    np.testing.assert_allclose(delayed[0, lag:], baseline[0, :-lag], atol=1e-6)


def test_negative_impulse_offset_keeps_a_sum_normalized_kernel_normalized():
    """Test that `norm="sum"` still sums to one, since the leading zeros add nothing.

    The normalizer is not identical to the unshifted one -- the truncated tail is missing from it -- so
    a sum-normalized kernel is rescaled slightly rather than merely shifted.

    """
    params = pd.DataFrame(index=range(1))

    resp = np.asarray(TwoGammaImpulse(offset=-5.0)(params))

    np.testing.assert_allclose(resp.sum(axis=1), 1.0, rtol=1e-5)
