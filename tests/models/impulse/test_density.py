"""Tests for density functions."""

from itertools import product
import numpy as np
import pytest
from scipy import differentiate
from scipy import integrate
from scipy import special
from scipy import stats
from prfmodel.models.impulse import derivative_gamma_density
from prfmodel.models.impulse import gamma_density
from prfmodel.models.impulse import shifted_derivative_gamma_density
from prfmodel.models.impulse import shifted_gamma_density


class TestGammaDensitySetup:
    """Setup for testing gamma density functions."""

    duration = 32
    offset = 0.0001
    resolution = 0.1

    @pytest.fixture
    def frames(self):
        """Time frames."""
        # Frames must have shape (n, 1)
        return np.expand_dims(np.linspace(self.offset, self.duration, int(self.duration / self.resolution)), 0)

    @pytest.fixture
    def parameter_range(self):
        """Range of shape and rate parameters."""
        return np.round(np.linspace(0.1, 5.0, 5), 2)

    def _check_peak(
        self,
        frames: np.ndarray,
        response: np.ndarray,
        expected_mode: np.ndarray,
        shift: np.ndarray = None,
    ) -> None:
        frames = frames.squeeze()
        # Get maximum of response
        peak_frame_idx = np.argmax(response, axis=1)
        # Get time frame with maximum
        peak_response = frames[peak_frame_idx]

        shift = [None] * len(expected_mode) if shift is None else shift.squeeze()

        for ep, pr, idx, sh in zip(expected_mode, peak_response, peak_frame_idx, shift, strict=False):
            # Expected mode is later than time frames
            if ep >= frames.max():
                assert idx == (len(frames) - 1), "Peak response must be in last frame"
            # Expected mode is earlier than time frames
            elif ep <= frames.min():
                if sh is None:
                    assert idx == 0, "Peak response must be in first frame"
                else:
                    first_nonzero_idx = np.argmax(frames > sh)
                    assert idx == first_nonzero_idx, "Peak response must be in first nonzero frame"
            else:
                # The observed peak should not differ from expected peak more than the resolution
                assert abs(pr - ep) <= self.resolution


class TestGammaDensity(TestGammaDensitySetup):
    """Tests for gamma_density function."""

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray):
        """Shape and rate parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range)))

    def test_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that gamma density peaks at correct frame across combinations of shape and rate parameters.

        Argument `parameters` is a two-dimensional array where the first column is the shape and the second column
        the rate parameter of each parameter combination.

        The peak of the gamma density is tested against the expected analytical mode of the gamma distribution.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)

        resp = np.asarray(gamma_density(frames, shape, rate))

        assert np.all(resp > 0.0)

        # Calc expected analytical mode of each parameter combination
        expected_mode = np.where(shape < 1, 0, (shape - 1) / rate).squeeze()

        self._check_peak(frames, resp, expected_mode)

    def test_gamma_density_integral(self):
        """Test that the integral of normalized density is 1."""
        integ = integrate.quad(gamma_density, 0, np.inf, args=(2.0, 1.0, True))

        assert integ[0] == pytest.approx(1.0)

    def test_gamma_density_unnormalized(self, frames: np.ndarray):
        """Test that the normalized density is equal to the unnormalized density times the normalizing constant."""
        shape = np.array([[2.0]])
        rate = np.array([[1.0]])
        dens_norm = np.asarray(gamma_density(frames, shape, rate))
        dens_unnorm = np.asarray(gamma_density(frames, shape, rate, norm=False))

        assert np.all(dens_norm == dens_unnorm * (rate**shape / special.gamma(shape)))

    def test_values_value_error(self, frames: np.ndarray):
        """Test that negative values raise an error."""
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])
        frames[0] = -1.0

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_shape_value_error(self, frames: np.ndarray):
        """Test that negative shape parameters raise an error."""
        shape = np.array([[-1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_rate_value_error(self, frames: np.ndarray):
        """Test that negative rate parameters raise an error."""
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[-1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)


class TestShiftedGammaDensity(TestGammaDensitySetup):
    """Tests for shifted_gamma_density function."""

    @pytest.fixture
    def shift_parameter_range(self):
        """Range for shift parameter."""
        return np.linspace(-5, 5, num=5)

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray, shift_parameter_range: np.ndarray):
        """Shape, rate, and shift parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range, shift_parameter_range)))

    def test_shifted_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that shifted gamma density peaks at correct frame across combinations of shape, rate, and shift parameters.

        Argument `parameters` is a three-dimensional array where the first column is the shape, the second column
        the rate parameter, and the third columnt the shift parameter of each parameter combination.

        The peak of the gamma density is tested against the shifted expected analytical mode of the gamma distribution.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)
        shift = np.expand_dims(parameters[:, 2], 1)

        resp = np.asarray(shifted_gamma_density(frames, shape, rate, shift))

        assert np.all(resp >= 0.0)

        # Calc expected analytical mode of each parameter combination
        expected_mode = (np.where(shape < 1, 0, (shape - 1) / rate) + shift).squeeze()

        self._check_peak(frames, resp, expected_mode, shift)


class TestDerivativeGammaDensity(TestGammaDensitySetup):
    """Tests for derivative_gamma_density function."""

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray):
        """Shape and rate parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range)))

    def test_derivative_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """Test that the derivative gamma density is close to the approximate derivative from scipy."""
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)

        frames = frames[:, 1:]  # Don't compare at first frame because non-analytic derivativ is not stable

        frames = frames.squeeze()

        resp = np.asarray(derivative_gamma_density(frames, shape, rate))

        # Calc the approximate derivative for each parameter combination
        ref = np.array(
            [
                differentiate.derivative(
                    lambda x, p=p: stats.gamma.pdf(x, a=p[0], scale=1 / p[1]),
                    frames,
                ).df
                for p in parameters
            ],
        )

        assert np.all(np.isclose(resp, ref))


class TestShiftedDerivativeGammaDensity(TestGammaDensitySetup):
    """Tests for shifted_derivative_gamma_density function."""

    @pytest.fixture
    def parameter_range(self):
        """
        Range of shape and rate parameters.

        We only test for values >= 2 because the derivative behaves well in this range.

        """
        return np.round(np.linspace(2.0, 5.0, 5), 2)

    @pytest.fixture
    def shift_parameter_range(self):
        """
        Range for shift parameter.

        We only test for values >= 0 because the derivative behaves well in this range.

        """
        return np.linspace(0.0, 5.0, num=5)

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray, shift_parameter_range: np.ndarray):
        """Shape, rate, and shift parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range, shift_parameter_range)))

    def test_shifted_derivative_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that shifted gamma density peaks at correct frame across combinations of shape, rate, and shift parameters.

        Argument `parameters` is a three-dimensional array where the first column is the shape, the second column
        the rate parameter, and the third columnt the shift parameter of each parameter combination.

        The peak of the gamma density is tested against the shifted expected analytical mode of the gamma distribution.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)
        shift = np.expand_dims(parameters[:, 2], 1)

        resp = np.asarray(shifted_derivative_gamma_density(frames, shape, rate, shift))

        # Calc expected analytical mode of each parameter combination
        expected_mode = ((shape - 1 - np.sqrt(shape - 1)) / rate + shift).squeeze()

        self._check_peak(frames, resp, expected_mode, shift)
