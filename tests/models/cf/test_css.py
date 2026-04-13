"""Tests for compressive spatial summation CF model class."""

import pandas as pd
import pytest
from pytest_regressions.num_regression import NumericRegressionFixture
from prfmodel.models.cf.gaussian import GaussianCFModel
from prfmodel.models.cf.stimulus_encoding import CFStimulusEncoder
from prfmodel.models.compression import CompressiveEncoder
from prfmodel.stimuli.cf import CFStimulus
from tests.models.conftest import CFSetup


class TestCompressiveGaussianCFModel(CFSetup):
    """Tests for compressive spatial summation Gaussian CF model."""

    @pytest.fixture
    def cf_model(self):
        """CF model object."""
        return GaussianCFModel(
            encoding_model=CompressiveEncoder(CFStimulusEncoder()),
        )

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "center_index": [0, 2, 1],
                "sigma": [1.0, 2.0, 3.0],
                "baseline": [0.5, -0.1, 0.2],
                "amplitude": [-1.1, 0.5, 2.0],
                "gain": [0.1, 0.2, 0.3],
                "n": [0.9, 1.1, 0.1],
            },
        )

    def test_predict(
        self,
        cf_model: GaussianCFModel,
        stimulus: CFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape."""
        resp = cf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.source_response.shape[-1])

    def test_predict_regression_cf_css(
        self,
        cf_model: GaussianCFModel,
        num_regression: NumericRegressionFixture,
        stimulus: CFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference file."""
        resp = cf_model(stimulus, params)

        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4},
        )
