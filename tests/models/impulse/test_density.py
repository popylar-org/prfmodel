"""Tests for density functions."""

from functools import partial
from itertools import product
import numpy as np
import pytest
from scipy import differentiate
from scipy import integrate
from scipy import special
from scipy import stats
from prfmodel.models.base import BatchDimensionError
from prfmodel.models.impulse import derivative_gamma_density
from prfmodel.models.impulse import gamma_density
from prfmodel.models.impulse import shifted_gamma_density


class TestGammaDensitySetup:
    """Setup for testing gamma density functions."""

    duration = 32.0
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

    @pytest.fixture
    def shift_parameter_range(self):
        """Range for shift parameter."""
        return np.linspace(-5, 5, num=5)

    @staticmethod
    def _calc_gamma_pdf(x: np.ndarray, shape: float, rate: float, shift: float = 0.0) -> np.ndarray:
        return stats.gamma.pdf(x, a=shape, loc=shift, scale=1 / rate)


class TestGammaDensity(TestGammaDensitySetup):
    """Tests for gamma_density function."""

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray):
        """Shape and rate parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range)))

    def test_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that gamma density is the same as `scipy.stats.gamma.pdf`.

        Argument `parameters` is a two-dimensional array where the first column is the shape and the second column
        the rate parameter of each parameter combination.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)

        resp = np.asarray(gamma_density(frames, shape, rate))

        ref = self._calc_gamma_pdf(frames, shape, rate)

        assert np.all(np.isclose(resp, ref))

    def test_gamma_density_scalar(self):
        """Test that gamma density is the same as `scipy.stats.gamma.pdf` for scalar inputs."""
        frames = 1.0
        shape = 2.0
        rate = 1.0

        resp = np.asarray(gamma_density(frames, shape, rate))

        ref = self._calc_gamma_pdf(frames, shape, rate)

        assert np.all(np.isclose(resp, ref))

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

    def test_values_shape_value_error(self):
        """Test that values with the wrong shape raise an error."""
        frames = np.ones((3,))
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

        frames = np.ones((3, 1))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

        frames = np.ones((1, 3, 1))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_values_negative_value_error(self, frames: np.ndarray):
        """Test that negative values raise an error."""
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])
        frames[0] = -1.0

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_shape_shape_value_error(self, frames: np.ndarray):
        """Test that shape parameters with the wrong shape raise an error."""
        shape = np.ones((3,))
        rate = np.ones((3, 1))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

        shape = np.ones((1, 3))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

        shape = np.ones((3, 1, 1))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_shape_negative_value_error(self, frames: np.ndarray):
        """Test that negative shape parameters raise an error."""
        shape = np.array([[-1.0, 2.0]])
        rate = np.array([[1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_rate_shape_value_error(self, frames: np.ndarray):
        """Test that rate parameters with the wrong shape raise an error."""
        shape = np.ones((3, 1))
        rate = np.ones((3,))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

        rate = np.ones((1, 3))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

        rate = np.ones((3, 1, 1))

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_rate_negative_value_error(self, frames: np.ndarray):
        """Test that negative rate parameters raise an error."""
        shape = np.array([[1.0, 2.0]])
        rate = np.array([[-1.0, 1.0]])

        with pytest.raises(ValueError):
            gamma_density(frames, shape, rate)

    def test_heterogeneous_shape_error(self, frames: np.ndarray):
        """Test that parameters with different shapes raise an error."""
        shape = np.ones((3, 1))
        rate = np.ones((2, 1))

        with pytest.raises(BatchDimensionError):
            gamma_density(frames, shape, rate)


class TestShiftedGammaDensity(TestGammaDensitySetup):
    """Tests for shifted_gamma_density function."""

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray, shift_parameter_range: np.ndarray):
        """Shape, rate, and shift parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range, shift_parameter_range)))

    def test_shifted_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that shifted gamma density is the same as `scipy.stats.gamma.pdf`.

        Argument `parameters` is a three-dimensional array where the first column is the shape, the second column
        the rate parameter, and the third columnt the shift parameter of each parameter combination.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)
        shift = np.expand_dims(parameters[:, 2], 1)

        resp = np.asarray(shifted_gamma_density(frames, shape, rate, shift))

        ref = self._calc_gamma_pdf(frames, shape, rate, shift)

        assert np.all(np.isclose(resp, ref))

    def test_shifted_gamma_density_scalar(self):
        """Test that shifted gamma density is the same as `scipy.stats.gamma.pdf` for scalar inputs."""
        frames = 1.0
        shape = 2.0
        rate = 1.0
        shift = 2.0

        resp = np.asarray(shifted_gamma_density(frames, shape, rate, shift))

        ref = self._calc_gamma_pdf(frames, shape, rate, shift)

        assert np.all(np.isclose(resp, ref))

    def test_shift_shape_value_error(self, frames: np.ndarray):
        """Test that shift parameters with the wrong shape raise an error."""
        shape = np.ones((3, 1))
        rate = np.ones((3, 1))
        shift = np.ones((3,))

        with pytest.raises(ValueError):
            shifted_gamma_density(frames, shape, rate, shift)

        shift = np.ones((1, 3))

        with pytest.raises(ValueError):
            shifted_gamma_density(frames, shape, rate, shift)

        shift = np.ones((3, 1, 1))

        with pytest.raises(ValueError):
            shifted_gamma_density(frames, shape, rate, shift)

    def test_heterogeneous_shape_error(self, frames: np.ndarray):
        """Test that parameters with different shapes raise an error."""
        shape = np.ones((3, 1))
        shape = np.ones((3, 1))
        rate = np.ones((3, 1))
        shift = np.ones((2, 1))

        with pytest.raises(BatchDimensionError):
            shifted_gamma_density(frames, shape, rate, shift)


class TestDerivativeGammaDensity(TestGammaDensitySetup):
    """Tests for derivative_gamma_density function."""

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray):
        """Shape and rate parameter combinations."""
        return np.array(list(product(parameter_range, parameter_range)))

    def test_derivative_gamma_density(self, frames: np.ndarray, parameters: np.ndarray):
        """
        Test that the derivative gamma density is close to the approximate derivative from scipy.

        Note that the approximate derivative is unstable at the first time frame, so we omit it for testing.

        """
        # Parameters must have shape (n, 1)
        shape = np.expand_dims(parameters[:, 0], 1)
        rate = np.expand_dims(parameters[:, 1], 1)

        frames = frames[:, 1:]  # Don't compare at first frame because non-analytic derivative is not stable

        resp = np.asarray(derivative_gamma_density(frames, shape, rate)).squeeze()

        frames = frames.squeeze()  # Omit first dimension for approximate derivative

        # Calc the approximate derivative for each parameter combination
        ref = np.array(
            [
                differentiate.derivative(partial(self._calc_gamma_pdf, shape=p[0], rate=p[1]), frames).df
                for p in parameters
            ],
        )

        assert np.all(np.isclose(resp, ref))
