"""Tests for temporal model classes."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.base import ShapeError
from prfmodel.models.temporal import BaselineAmplitude
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
