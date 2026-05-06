"""Tests for temporal model classes."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.base import ShapeError
from prfmodel.models.temporal import BaselineAmplitude
from prfmodel.models.temporal import DivNormAmplitude
from prfmodel.models.temporal import DoGAmplitude
from .conftest import parametrize_dtype


class TestBaselineAmplitdue:
    """Tests for BaselineAmplitude class."""

    num_frames = 10

    @pytest.fixture
    def model(self):
        """Model object."""
        return BaselineAmplitude()

    @pytest.fixture
    def params(self):
        """Model parameters."""
        return pd.DataFrame(
            {
                "baseline": [5.0, 10.0, -3.0],
                "amplitude": [2.0, -1.0, 1.0],
            },
        )

    @parametrize_dtype
    def test_call(self, model: BaselineAmplitude, params: pd.DataFrame, dtype: str):
        """Test that BaselineAmplitude returns response with correct shape."""
        inputs = np.ones((params.shape[0], self.num_frames))

        resp = model(inputs, params, dtype)

        assert resp.shape == inputs.shape
        assert np.allclose(
            resp,
            inputs * np.expand_dims(params["amplitude"], 1) + np.expand_dims(params["baseline"], 1),
        )

    def test_shape_error(self, model: BaselineAmplitude, params: pd.DataFrame):
        """Test that ShapeError is raised."""
        inputs = np.ones(self.num_frames)

        with pytest.raises(ShapeError):
            model(inputs, params)


class TestDoGAmplitude:
    """Tests for DoGAmplitude class."""

    num_frames = 10

    @pytest.fixture
    def model(self):
        """Model object."""
        return DoGAmplitude()

    @pytest.fixture
    def params(self):
        """Model parameters."""
        return pd.DataFrame(
            {
                "amplitude_center": [2.0, -1.0, 1.0],
                "amplitude_surround": [0.5, 0.3, -0.5],
                "baseline": [5.0, 10.0, -3.0],
            },
        )

    def test_parameter_names(self, model: DoGAmplitude):
        """Test that correct parameter names are returned."""
        assert model.parameter_names == ["amplitude_center", "amplitude_surround", "baseline"]

    @parametrize_dtype
    def test_call(self, model: DoGAmplitude, params: pd.DataFrame, dtype: str):
        """Test that DoGAmplitude returns response with correct shape and values."""
        num_units = params.shape[0]
        p1 = np.ones((num_units, self.num_frames)) * 2.0
        p2 = np.ones((num_units, self.num_frames)) * 3.0
        inputs = np.stack([p1, p2], axis=1)  # (num_units, 2, num_frames)

        resp = np.asarray(model(inputs, params, dtype))

        assert resp.shape == (num_units, self.num_frames)
        expected = (
            p1 * np.expand_dims(params["amplitude_center"], 1)
            + p2 * np.expand_dims(params["amplitude_surround"], 1)
            + np.expand_dims(params["baseline"], 1)
        )
        assert np.allclose(resp, expected)

    def test_shape_error(self, model: DoGAmplitude, params: pd.DataFrame):
        """Test that ShapeError is raised for wrong number of dimensions."""
        inputs = np.ones(self.num_frames)

        with pytest.raises(ShapeError):
            model(inputs, params)


class TestDivNormAmplitude:
    """Tests for DivNormAmplitude class."""

    num_frames = 10

    @pytest.fixture
    def model(self):
        """Model object."""
        return DivNormAmplitude()

    @pytest.fixture
    def params(self):
        """Model parameters."""
        return pd.DataFrame(
            {
                "amplitude_activation": [2.0, -1.0, 1.0],
                "baseline_activation": [0.5, 0.0, 1.0],
                "amplitude_normalization": [1.0, 0.5, 2.0],
                "baseline_normalization": [1.0, 2.0, 0.5],
            },
        )

    def test_parameter_names(self, model: DivNormAmplitude):
        """Test that correct parameter names are returned."""
        assert model.parameter_names == [
            "amplitude_activation",
            "baseline_activation",
            "amplitude_normalization",
            "baseline_normalization",
        ]

    @parametrize_dtype
    def test_call(self, model: DivNormAmplitude, params: pd.DataFrame, dtype: str):
        """Test that DivNormAmplitude returns correct shape and values."""
        num_voxels = params.shape[0]
        p1 = np.ones((num_voxels, self.num_frames)) * 2.0
        p2 = np.ones((num_voxels, self.num_frames)) * 3.0
        inputs = np.stack([p1, p2], axis=1)

        resp = np.asarray(model(inputs, params, dtype))

        assert resp.shape == (num_voxels, self.num_frames)

        a = np.expand_dims(params["amplitude_activation"].to_numpy(), 1)
        b = np.expand_dims(params["baseline_activation"].to_numpy(), 1)
        c = np.expand_dims(params["amplitude_normalization"].to_numpy(), 1)
        d = np.expand_dims(params["baseline_normalization"].to_numpy(), 1)
        expected = (a * p1 + b) / (c * p2 + d) - b / d
        assert np.allclose(resp, expected)

    @parametrize_dtype
    def test_call_no_subtract_baseline(self, params: pd.DataFrame, dtype: str):
        """Test that subtract_baseline=False omits the b/d correction term."""
        model = DivNormAmplitude(subtract_baseline=False)
        num_voxels = params.shape[0]
        p1 = np.ones((num_voxels, self.num_frames)) * 2.0
        p2 = np.ones((num_voxels, self.num_frames)) * 3.0
        inputs = np.stack([p1, p2], axis=1)

        resp = np.asarray(model(inputs, params, dtype))

        a = np.expand_dims(params["amplitude_activation"].to_numpy(), 1)
        b = np.expand_dims(params["baseline_activation"].to_numpy(), 1)
        c = np.expand_dims(params["amplitude_normalization"].to_numpy(), 1)
        d = np.expand_dims(params["baseline_normalization"].to_numpy(), 1)
        expected = (a * p1 + b) / (c * p2 + d)
        assert np.allclose(resp, expected)

    def test_zero_response_no_stimulus(self, model: DivNormAmplitude, params: pd.DataFrame):
        """Test that the response is zero when there is no stimulus (p1 = p2 = 0)."""
        num_voxels = params.shape[0]
        inputs = np.zeros((num_voxels, 2, self.num_frames))

        resp = np.asarray(model(inputs, params))

        assert np.allclose(resp, 0.0)

    def test_shape_error(self, model: DivNormAmplitude, params: pd.DataFrame):
        """Test that ShapeError is raised for wrong number of dimensions."""
        inputs = np.ones(self.num_frames)

        with pytest.raises(ShapeError):
            model(inputs, params)
