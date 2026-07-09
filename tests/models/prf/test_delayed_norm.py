"""Tests for DelayedNormGaussian2DPRFModel and init_delayed_norm_from_gaussian."""

import pandas as pd
import pytest
from pytest_regressions.num_regression import NumericRegressionFixture
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.prf import DelayedNormGaussian2DPRFModel
from prfmodel.models.prf import init_delayed_norm_from_gaussian
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup


class TestDelayedNormGaussian2DPRFModel(PRFStimulusSetup):
    """Tests for DelayedNormGaussian2DPRFModel."""

    @pytest.fixture
    def prf_model(self):
        """Model object."""
        return DelayedNormGaussian2DPRFModel()

    @pytest.fixture
    def params(self):
        """Parameter DataFrame with paper default values."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 1.5, 2.0],
                "delay": [0.05, 0.05, 0.05],
                "dispersion": [0.9, 0.9, 0.9],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.0, 0.0, 0.0],
                "n": [2.0, 2.0, 2.0],
                "dispersion_normalization": [0.1, 0.1, 0.1],
                "sigma_saturation": [1.0, 1.0, 1.0],
                "amplitude": [1.0, 1.0, 1.0],
                "baseline": [0.0, 0.0, 0.0],
            },
        )

    def test_parameter_names(self, prf_model: DelayedNormGaussian2DPRFModel):
        """Test that parameter_names contains all expected parameters in order."""
        expected = [
            "mu_y",
            "mu_x",
            "sigma",
            "delay",
            "dispersion",
            "undershoot",
            "u_dispersion",
            "ratio",
            "weight_deriv",
            "n",
            "dispersion_normalization",
            "sigma_saturation",
            "baseline",
            "amplitude",
        ]
        assert prf_model.parameter_names == expected

    @pytest.mark.parametrize("impulse_model", [None, DerivativeTwoGammaImpulse(), DerivativeTwoGammaImpulse])
    def test_predict_shape_parametrized(
        self,
        impulse_model: BaseImpulse,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test output shape for various impulse model combinations."""
        model = DelayedNormGaussian2DPRFModel(impulse_model=impulse_model)
        resp = model(stimulus, params)
        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    @pytest.mark.parametrize("impulse_model", [None, DerivativeTwoGammaImpulse()])
    def test_predict_regression(
        self,
        num_regression: NumericRegressionFixture,
        impulse_model: BaseImpulse,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference values."""
        model = DelayedNormGaussian2DPRFModel(impulse_model=impulse_model)
        resp = model(stimulus, params)
        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4, "rtol": 1e-3},
        )


class TestInitDelayedNormFromGaussian:
    """Tests for init_delayed_norm_from_gaussian."""

    @pytest.fixture
    def gaussian_params(self):
        """Gaussian model parameters."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0],
                "mu_y": [0.0, -1.0],
                "sigma": [1.0, 2.0],
                "delay": [0.05, 0.05],
                "dispersion": [0.9, 0.9],
                "undershoot": [12.0, 12.0],
                "u_dispersion": [0.9, 0.9],
                "ratio": [0.48, 0.48],
                "weight_deriv": [0.0, 0.0],
                "baseline": [0.0, 0.1],
                "amplitude": [1.0, -1.0],
            },
        )

    def test_output_columns(self, gaussian_params: pd.DataFrame):
        """Result has all original columns plus the three new DGN parameters."""
        dgn_params = init_delayed_norm_from_gaussian(gaussian_params)
        new_cols = {"n", "dispersion_normalization", "sigma_saturation"}
        assert new_cols.issubset(set(dgn_params.columns))

    def test_passthrough_columns(self, gaussian_params: pd.DataFrame):
        """All original columns pass through unchanged, including scaling parameters."""
        dgn_params = init_delayed_norm_from_gaussian(gaussian_params)
        for col in ["mu_x", "mu_y", "sigma", "delay", "dispersion", "amplitude", "baseline"]:
            assert list(dgn_params[col]) == list(gaussian_params[col])

    def test_paper_defaults(self, gaussian_params: pd.DataFrame):
        """Default arguments match paper-recommended values."""
        dgn_params = init_delayed_norm_from_gaussian(gaussian_params)
        assert list(dgn_params["n"]) == [2.0, 2.0]
        assert list(dgn_params["dispersion_normalization"]) == [0.1, 0.1]
        assert list(dgn_params["sigma_saturation"]) == [1.0, 1.0]

    def test_custom_values(self, gaussian_params: pd.DataFrame):
        """Custom scalar arguments are broadcast to all rows."""
        dgn_params = init_delayed_norm_from_gaussian(
            gaussian_params,
            n=3.0,
            dispersion_normalization=0.2,
            sigma_saturation=0.5,
        )
        assert list(dgn_params["n"]) == [3.0, 3.0]
        assert list(dgn_params["dispersion_normalization"]) == [0.2, 0.2]
        assert list(dgn_params["sigma_saturation"]) == [0.5, 0.5]
