"""Tests for compressive spatial summation pRF model class."""

import pandas as pd
import pytest
from pytest_regressions.num_regression import NumericRegressionFixture
from prfmodel.models.prf.css import Gaussian2DCSSPRFModel
from prfmodel.models.prf.gaussian import Gaussian2DPRFModel
from prfmodel.stimuli.prf import PRFStimulus
from tests.conftest import PRFStimulusSetup


class TestCompressiveGaussian2DPRFModel(PRFStimulusSetup):
    """Tests for compressive spatial summation 2D Gaussian PRF model."""

    @pytest.fixture
    def css_model(self):
        """PRF model object."""
        return Gaussian2DCSSPRFModel()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "baseline": [0.0, 0.1, 0.2],
                "amplitude": [1.1, 1.0, 0.9],
                "gain": [0.1, 0.2, 0.3],
                "n": [0.9, 1.1, 0.1],
            },
        )

    def test_predict(
        self,
        css_model: Gaussian2DPRFModel,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape."""
        resp = css_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    def test_predict_regression_css(
        self,
        num_regression: NumericRegressionFixture,
        css_model: Gaussian2DPRFModel,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference file."""
        resp = css_model(stimulus, params)

        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4},
        )
