"""Tests for utility functions and classes."""

from collections.abc import Callable
import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import _get_norm_fun
from prfmodel.utils import batched
from prfmodel.utils import calculate_eccentricity
from prfmodel.utils import calculate_polar_angle
from prfmodel.utils import calculate_r_squared
from prfmodel.utils import normalize_response
from .conftest import TestSetup


@pytest.mark.parametrize("norm", [None, "sum", "mean", "max", "norm"])
def test_normalize_response(norm: str):
    """Test that normalize_response returns correct result."""
    response = np.expand_dims(np.linspace(-5, 5, 100), 0)
    response_norm = np.asarray(normalize_response(response, norm=norm))

    assert response_norm.shape == response.shape

    if norm is not None:
        norm_fun = _get_norm_fun(norm)
        response_norm_ref = response / np.asarray(norm_fun(response, axis=1, keepdims=True))
    else:
        response_norm_ref = response

    assert np.allclose(response_norm, response_norm_ref)


def test_normalize_response_error():
    """Test that normalize_response raises an error for wrong input shape."""
    response = np.ones((10,))

    with pytest.raises(ValueError):
        normalize_response(response)

    response = 10

    with pytest.raises(ValueError):
        normalize_response(response)

    response = np.ones((10, 2, 1))

    with pytest.raises(ValueError):
        normalize_response(response)


@pytest.mark.parametrize(
    ("mu_x", "mu_y", "expected"),
    [
        (1.0, 0.0, 0.0),  # Right
        (1.0, 1.0, np.pi / 4),  # Upper right
        (0.0, 1.0, np.pi / 2),  # Up
        (-1.0, 1.0, 3 * np.pi / 4),  # Upper left
        (-1.0, 0.0, np.pi),  # Left
        (-1.0, -1.0, -3 * np.pi / 4),  # Lower left
        (0.0, -1.0, -np.pi / 2),  # Down
        (1.0, -1.0, -np.pi / 4),  # Lower right
    ],
)
def test_calculate_polar_angle(mu_x: float, mu_y: float, expected: float):
    """Test that the polar angle runs counterclockwise from the positive x-axis in all quadrants."""
    angle = calculate_polar_angle(np.array([mu_x]), np.array([mu_y]))
    np.testing.assert_allclose(angle, [expected], atol=1e-12)


def test_calculate_polar_angle_scale_invariant():
    """Test that the polar angle does not depend on the distance of the pRF center from the origin."""
    mu_x = np.array([1.0, -2.0, 0.5])
    mu_y = np.array([3.0, 1.0, -4.0])
    np.testing.assert_allclose(calculate_polar_angle(mu_x, mu_y), calculate_polar_angle(10.0 * mu_x, 10.0 * mu_y))


def test_calculate_polar_angle_range():
    """Test that the polar angle lies in [-pi, pi]."""
    rng = np.random.default_rng(0)
    angle = calculate_polar_angle(rng.normal(size=1000), rng.normal(size=1000))
    assert np.all(angle >= -np.pi)
    assert np.all(angle <= np.pi)


@pytest.mark.parametrize(
    ("mu_x", "mu_y", "expected"),
    [
        (0.0, 0.0, 0.0),
        (3.0, 4.0, 5.0),
        (-3.0, 4.0, 5.0),
        (3.0, -4.0, 5.0),
        (-1.0, 0.0, 1.0),
        (0.0, -2.0, 2.0),
    ],
)
def test_calculate_eccentricity(mu_x: float, mu_y: float, expected: float):
    """Test that the eccentricity is the distance of the pRF center from the origin."""
    np.testing.assert_allclose(calculate_eccentricity(np.array([mu_x]), np.array([mu_y])), [expected])


def test_calculate_polar_angle_eccentricity_round_trip():
    """Test that polar angle and eccentricity recover the pRF center coordinates."""
    rng = np.random.default_rng(0)
    mu_x = rng.normal(size=100)
    mu_y = rng.normal(size=100)
    angle = calculate_polar_angle(mu_x, mu_y)
    eccentricity = calculate_eccentricity(mu_x, mu_y)
    np.testing.assert_allclose(eccentricity * np.cos(angle), mu_x)
    np.testing.assert_allclose(eccentricity * np.sin(angle), mu_y)


@pytest.mark.parametrize("fn", [calculate_polar_angle, calculate_eccentricity])
def test_calculate_center_series_input(fn: Callable):
    """Test that pandas Series (e.g., parameter DataFrame columns) are accepted and return a numpy array."""
    parameters = pd.DataFrame({"mu_x": [1.0, 0.0], "mu_y": [0.0, 2.0]})
    result = fn(parameters["mu_x"], parameters["mu_y"])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, fn(parameters["mu_x"].to_numpy(), parameters["mu_y"].to_numpy()))


@pytest.mark.parametrize("fn", [calculate_polar_angle, calculate_eccentricity])
def test_calculate_center_nan(fn: Callable):
    """Test that NaN coordinates (e.g., of excluded units) propagate to the result."""
    result = fn(np.array([np.nan, 1.0, 1.0]), np.array([1.0, np.nan, 1.0]))
    assert np.isnan(result[:2]).all()
    assert np.isfinite(result[2])


@pytest.mark.parametrize("fn", [calculate_polar_angle, calculate_eccentricity])
def test_calculate_center_shape(fn: Callable):
    """Test that the result keeps the shape of the inputs (e.g., values projected onto a surface)."""
    rng = np.random.default_rng(0)
    assert fn(rng.normal(size=(4, 5)), rng.normal(size=(4, 5))).shape == (4, 5)


def test_calculate_r_squared_matches_keras_metric():
    """Test that the R-squared matches the per-unit score of the Keras R2Score metric."""
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(20, 50))
    predicted = observed + rng.normal(scale=0.5, size=(20, 50))
    # The Keras metric computes the score along the first axis, so units must be on the second axis
    expected = keras.metrics.R2Score(class_aggregation=None, dtype="float64")(observed.T, predicted.T)
    np.testing.assert_allclose(calculate_r_squared(observed, predicted), keras.ops.convert_to_numpy(expected))


def test_calculate_r_squared_values():
    """Test the R-squared of perfect, mean, and anticorrelated predictions."""
    observed = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
    predicted = np.stack([observed[0], np.full(4, 2.5), observed[0][::-1]])
    np.testing.assert_allclose(calculate_r_squared(observed, predicted), [1.0, 0.0, -3.0])


def test_calculate_r_squared_tensor_input():
    """Test that backend tensors are accepted and a numpy array is returned."""
    observed = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
    result = calculate_r_squared(keras.ops.convert_to_tensor(observed), keras.ops.convert_to_tensor(observed))
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, [1.0, 1.0])


def test_calculate_r_squared_single_unit():
    """Test that a single 1-dimensional timecourse gives a single score."""
    result = calculate_r_squared(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    assert result.shape == ()
    np.testing.assert_allclose(result, 0.5)


def test_calculate_r_squared_constant_observed():
    """Test that the R-squared is NaN for a constant observed response."""
    observed = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    result = calculate_r_squared(observed, np.zeros_like(observed))
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_calculate_r_squared_shape_mismatch():
    """Test that responses with different shapes raise an error."""
    with pytest.raises(ValueError, match="same shape"):
        calculate_r_squared(np.zeros((2, 5)), np.zeros((2, 6)))


class TestTensorFrame:
    """Tests for TensorFrame class."""

    shape: tuple[int] = (3, 1)

    @pytest.fixture
    def tensor_frame(self):
        """TensorFrame object."""
        return TensorFrame({"a": 0.0, "b": [1.0], "c": np.ones(self.shape[0]), "d": keras.ops.ones(self.shape)})

    def test_get_item(self, tensor_frame: TensorFrame):
        """Test that getting an item with a single key returns the correct shape and values."""
        for key in tensor_frame.columns:
            x = tensor_frame[key]
            assert x.shape == self.shape[:1]

        # torch requires us to convert tensors to numpy arrays before we can compare against floats
        assert np.all(np.asarray(tensor_frame["a"]) == 0.0)
        assert np.all(np.asarray(tensor_frame["b"]) == 1.0)

    def test_get_item_list(self, tensor_frame: TensorFrame):
        """Test that getting items with a list of keys returns the correct shapes and values."""
        x = tensor_frame[tensor_frame.columns]
        assert x.shape == (self.shape[0], len(tensor_frame.columns))

    def test_set_item(self, tensor_frame: TensorFrame):
        """Test that setting an item with a single key stores the correct shape and values."""
        new_tensor_frame = TensorFrame(
            {
                "e": keras.ops.zeros(self.shape),
            },
        )

        for key in tensor_frame.columns:
            new_tensor_frame[key] = tensor_frame[key]
            x = new_tensor_frame[key]
            assert x.shape == self.shape[:1]

    def test_set_item_list(self, tensor_frame: TensorFrame):
        """Test that setting items with a list of keys stores the correct shapes and values."""
        new_tensor_frame = TensorFrame(
            {
                "e": keras.ops.zeros(self.shape),
            },
        )
        new_tensor_frame[tensor_frame.columns] = tensor_frame[tensor_frame.columns]
        x = new_tensor_frame[tensor_frame.columns]
        assert x.shape == (self.shape[0], len(tensor_frame.columns))


class TestBatched(TestSetup):
    """Tests for the batched decorator."""

    def test_batch_size_none_returns_same_result(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that batch_size=None calls the function once with all units."""
        result_unbatched = model(stimulus, params)
        result_batched = batched(model)(stimulus, params, batch_size=None)

        assert np.array_equal(np.asarray(result_batched), np.asarray(result_unbatched))

    def test_batched_matches_unbatched(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that batched results match unbatched results."""
        result_unbatched = model(stimulus, params)
        result_batched = batched(model)(stimulus, params, batch_size=3)

        assert np.allclose(np.asarray(result_batched), np.asarray(result_unbatched))

    def test_output_shape(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that the output shape is (num_units, num_frames)."""
        result = batched(model)(stimulus, params, batch_size=3)

        assert result.shape == (params.shape[0], stimulus.design.shape[0])

    def test_batch_size_larger_than_num_units(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that a batch_size larger than the number of units works."""
        result = batched(model)(stimulus, params, batch_size=100)
        expected = model(stimulus, params)

        assert np.allclose(np.asarray(result), np.asarray(expected))

    def test_exact_batch_division(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test with a batch_size that evenly divides num_units."""
        result = batched(model)(stimulus, params, batch_size=3)
        expected = model(stimulus, params)

        assert np.allclose(np.asarray(result), np.asarray(expected))

    def test_passes_kwargs(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that keyword arguments are forwarded to the wrapped function."""
        expected_dtype = "float64"
        result = batched(model)(stimulus, params, batch_size=3, dtype=expected_dtype)

        assert keras.ops.dtype(result) == expected_dtype

    def test_decorator(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that the decorator syntax works with batch_size as a wrapper kwarg."""

        @batched
        def batched_call(stimulus: PRFStimulus, params: pd.DataFrame) -> Tensor:
            return model(stimulus, params)

        result = batched_call(stimulus, params, batch_size=3)
        expected = model(stimulus, params)

        assert np.allclose(np.asarray(result), np.asarray(expected))
