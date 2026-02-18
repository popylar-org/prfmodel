"""Tests for temporal model classes."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.base import ShapeError
from prfmodel.models.temporal import BaselineAmplitude
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
                "beta_1": [2.0, -1.0, 1.0],
                "beta_2": [0.5, 0.3, -0.5],
                "baseline": [5.0, 10.0, -3.0],
            },
        )

    def test_parameter_names(self, model: DoGAmplitude):
        """Test that correct parameter names are returned."""
        assert model.parameter_names == ["beta_1", "beta_2", "baseline"]

    @parametrize_dtype
    def test_call(self, model: DoGAmplitude, params: pd.DataFrame, dtype: str):
        """Test that DoGAmplitude returns response with correct shape and values."""
        num_voxels = params.shape[0]
        p1 = np.ones((num_voxels, self.num_frames)) * 2.0
        p2 = np.ones((num_voxels, self.num_frames)) * 3.0
        inputs = np.stack([p1, p2], axis=1)  # (num_voxels, 2, num_frames)

        resp = np.asarray(model(inputs, params, dtype))

        assert resp.shape == (num_voxels, self.num_frames)
        expected = (
            p1 * np.expand_dims(params["beta_1"], 1)
            + p2 * np.expand_dims(params["beta_2"], 1)
            + np.expand_dims(params["baseline"], 1)
        )
        assert np.allclose(resp, expected)

    def test_shape_error(self, model: DoGAmplitude, params: pd.DataFrame):
        """Test that ShapeError is raised for wrong number of dimensions."""
        inputs = np.ones(self.num_frames)

        with pytest.raises(ShapeError):
            model(inputs, params)
