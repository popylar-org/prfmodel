"""Test DifferenceOfGaussians2DResponse model class."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.difference_of_gaussians import DifferenceOfGaussians2DResponse
from prfmodel.stimulus import Stimulus
from tests.conftest import StimulusSetup
from .conftest import parametrize_dtype


class TestDifferenceOfGaussians2DResponse(StimulusSetup):
    """Tests for DifferenceOfGaussians2DResponse class."""

    @pytest.fixture
    def response_model(self):
        """Response model object."""
        return DifferenceOfGaussians2DResponse()

    def test_parameter_names(self, response_model: DifferenceOfGaussians2DResponse):
        """Test that correct parameter names are returned."""
        assert set(response_model.parameter_names) == {"mu_y", "mu_x", "sigma1", "sigma2"}

    @parametrize_dtype
    def test_predict(self, response_model: DifferenceOfGaussians2DResponse, stimulus: Stimulus, dtype: str):
        """Test that response prediction returns correct shape."""
        params = pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma1": [1.0, 1.5, 2.0],
                "sigma2": [2.0, 3.0, 4.0],
            },
        )

        preds = np.asarray(response_model(stimulus, params, dtype))

        assert preds.shape == (params.shape[0], stimulus.design.shape[1], stimulus.design.shape[2])

    def test_sigma2_less_than_sigma1_raises(self, response_model: DifferenceOfGaussians2DResponse, stimulus: Stimulus):
        """Test that ValueError is raised when sigma2 < sigma1."""
        params = pd.DataFrame(
            {
                "mu_x": [0.0],
                "mu_y": [0.0],
                "sigma1": [3.0],
                "sigma2": [1.0],
            },
        )

        with pytest.raises(ValueError, match="sigma2 must be greater than or equal to sigma1"):
            response_model(stimulus, params)

    def test_equal_sigmas_is_zero(self, response_model: DifferenceOfGaussians2DResponse, stimulus: Stimulus):
        """Test that equal sigma1 and sigma2 produces a zero response."""
        params = pd.DataFrame(
            {
                "mu_x": [0.0],
                "mu_y": [0.0],
                "sigma1": [2.0],
                "sigma2": [2.0],
            },
        )

        preds = np.asarray(response_model(stimulus, params))

        assert np.allclose(preds, 0.0)
