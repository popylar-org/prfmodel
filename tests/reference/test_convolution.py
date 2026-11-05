"""Tests that pin `convolve_prf_impulse_response` against `numpy.convolve`.

The convolution is the seam between the spatial and temporal halves of every pRF model, and nothing
else in the suite anchors it: the impulse tests cover the kernel, the model tests cover the encoded
response, and the regression snapshots cover their product without saying what that product should
be.

It also makes a choice that diverges from prfpy and braincoder, which zero-pad: the response is
padded on the left by repeating its first sample. That is deliberate -- it holds the pre-stimulus
signal at a steady state rather than ramping it up from zero -- but it fabricates history that was
never measured when frame 0 is non-zero, so it is asserted here rather than left implicit.

"""

import numpy as np
import pytest
from prfmodel.impulse._convolve import convolve_prf_impulse_response
from tests.reference._oracle import convolve


@pytest.fixture
def response() -> np.ndarray:
    """Return a single-unit response with a non-zero first sample, so padding choices are visible."""
    rng = np.random.default_rng(20260804)
    return rng.uniform(0.5, 2.0, size=24)


class TestDeltaKernels:
    """Tests the convolution against kernels whose effect is known exactly."""

    def test_unit_impulse_returns_response_unchanged(self, response: np.ndarray):
        """Test that convolving with a unit impulse is the identity."""
        kernel = np.zeros(8)
        kernel[0] = 1.0

        observed = np.asarray(
            convolve_prf_impulse_response(response[None, :], kernel[None, :], dtype="float64"),
        ).ravel()

        np.testing.assert_allclose(observed, response, rtol=1e-10)

    @pytest.mark.parametrize("delay", [1, 3, 7])
    def test_delayed_impulse_shifts_by_exactly_that_many_frames(self, response: np.ndarray, delay: int):
        """Test that an impulse at index k delays the response by k frames.

        This is what makes the impulse model's time axis meaningful: a kernel peaking at 5 s must
        move the response 5 s later. The leading `delay` samples come from the edge padding, so they
        are the first response value repeated.

        """
        kernel = np.zeros(12)
        kernel[delay] = 1.0

        observed = np.asarray(
            convolve_prf_impulse_response(response[None, :], kernel[None, :], dtype="float64"),
        ).ravel()

        expected = np.concatenate([np.full(delay, response[0]), response[:-delay]])

        np.testing.assert_allclose(observed, expected, rtol=1e-10)


class TestAgainstNumpy:
    """Tests the convolution against `numpy.convolve` for non-degenerate kernels."""

    def test_matches_numpy_convolve(self, response: np.ndarray):
        """Test that the depthwise-convolution implementation matches a plain NumPy convolution."""
        rng = np.random.default_rng(11)
        kernel = rng.uniform(0.0, 1.0, size=9)

        observed = np.asarray(
            convolve_prf_impulse_response(response[None, :], kernel[None, :], dtype="float64"),
        ).ravel()

        np.testing.assert_allclose(observed, convolve(response, kernel), rtol=1e-8)

    @pytest.mark.parametrize("num_units", [2, 3, 5])
    def test_units_are_convolved_independently(self, response: np.ndarray, num_units: int):
        """Test that unit `i` is convolved with kernel `i` and with no other unit's kernel.

        Every unit is given a different kernel, so a defect that pairs units with the wrong impulse
        response fails here. A shared kernel would hide it.

        """
        rng = np.random.default_rng(12)
        responses = rng.uniform(0.5, 2.0, size=(num_units, response.size))
        kernels = rng.uniform(0.0, 1.0, size=(num_units, 6))

        observed = np.asarray(convolve_prf_impulse_response(responses, kernels, dtype="float64"))

        expected = np.stack([convolve(responses[i], kernels[i]) for i in range(num_units)])

        np.testing.assert_allclose(observed, expected, rtol=1e-8)

    def test_batched_prediction_matches_predicting_each_unit_alone(self, response: np.ndarray):
        """Test that convolving a batch gives each unit what it would get on its own.

        Batching is an optimization, so it must never change a result. The single-unit path is the
        reference because it has no unit axis to get wrong.

        """
        rng = np.random.default_rng(13)
        responses = rng.uniform(0.5, 2.0, size=(3, response.size))
        kernels = rng.uniform(0.0, 1.0, size=(3, 6))

        batched = np.asarray(convolve_prf_impulse_response(responses, kernels, dtype="float64"))
        alone = np.stack(
            [
                np.asarray(
                    convolve_prf_impulse_response(responses[i][None, :], kernels[i][None, :], dtype="float64"),
                ).ravel()
                for i in range(3)
            ],
        )

        np.testing.assert_allclose(batched, alone, rtol=1e-8)
