"""Test TestDoG2DPRFModel model class."""

import pandas as pd
import pytest
from prfmodel.models.difference_of_gaussians import DoG2DPRFModel
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.temporal import DoGAmplitude
from prfmodel.stimulus import Stimulus
from tests.conftest import StimulusSetup


class TestDoG2DPRFModel(StimulusSetup):
    """Tests for the DoG2DPRFModel class."""

    @pytest.fixture
    def prf_model(self):
        """PRF model object."""
        return DoG2DPRFModel()

    @pytest.fixture
    def impulse_model(self):
        """Impulse response model object."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def temporal_model(self):
        """Temporal model object."""
        return DoGAmplitude()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma1": [1.0, 1.5, 2.0],
                "sigma2": [2.0, 3.0, 4.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "beta_1": [1.1, 1.0, 0.9],
                "beta_2": [-0.5, -0.3, -0.1],
                "baseline": [0.0, 0.1, 0.2],
            },
        )

    def test_parameter_names(
        self,
        prf_model: DoG2DPRFModel,
        impulse_model: DerivativeTwoGammaImpulse,
        temporal_model: DoGAmplitude,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        expected = ["mu_y", "mu_x", "sigma1", "sigma2"]
        expected.extend(impulse_model.parameter_names)
        expected.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(dict.fromkeys(expected))

    @pytest.mark.parametrize(
        ("impulse_model", "temporal_model"),
        [
            (DerivativeTwoGammaImpulse(), DoGAmplitude()),
            (DerivativeTwoGammaImpulse, DoGAmplitude),
            (None, None),
        ],
    )
    def test_predict(
        self,
        impulse_model,
        temporal_model,
        stimulus: Stimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape."""
        prf_model = DoG2DPRFModel(
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    def test_predict_responses(
        self,
        prf_model: DoG2DPRFModel,
        stimulus: Stimulus,
        params: pd.DataFrame,
    ):
        """Test that predict_responses returns stacked tensor with correct shape."""
        resp = prf_model.predict_responses(stimulus, params)

        assert resp.shape == (params.shape[0], 2, stimulus.design.shape[0])