"""Test TestDoG2DPRFModel model class."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.base import BaseImpulse
from prfmodel.models.base import BaseTemporal
from prfmodel.models.composite import CenterSurroundPRFModel
from prfmodel.models.difference_of_gaussians import DoG2DPRFModel
from prfmodel.models.difference_of_gaussians import init_dog_from_gaussian
from prfmodel.models.gaussian import Gaussian2DPRFResponse
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.temporal import DoGAmplitude
from prfmodel.stimuli.prf import PRFStimulus
from tests.conftest import PRFStimulusSetup


class TestDoG2DPRFModel(PRFStimulusSetup):
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
                "sigma_center": [1.0, 1.5, 2.0],
                "sigma_surround": [2.0, 3.0, 4.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "amplitude_center": [1.1, 1.0, 0.9],
                "amplitude_surround": [-0.5, -0.3, -0.1],
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
        expected = ["mu_y", "mu_x", "sigma_center", "sigma_surround"]
        expected.extend(impulse_model.parameter_names)
        expected.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(dict.fromkeys(expected))

    @pytest.mark.parametrize(
        ("impulse_model", "temporal_model"),
        [
            (DerivativeTwoGammaImpulse(), DoGAmplitude()),
            (DerivativeTwoGammaImpulse, DoGAmplitude),
            (DerivativeTwoGammaImpulse(), None),
            (None, DoGAmplitude()),
            (DerivativeTwoGammaImpulse, None),
            (None, DoGAmplitude),
            (None, None),
        ],
    )
    def test_predict(
        self,
        impulse_model: BaseImpulse,
        temporal_model: BaseTemporal,
        stimulus: PRFStimulus,
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
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that predict_responses returns stacked tensor with correct shape."""
        resp = prf_model.predict_responses(stimulus, params)

        assert resp.shape == (params.shape[0], 2, stimulus.design.shape[0])


class TestCenterSurroundPRFModel:
    """Tests for the change_params argument of CenterSurroundPRFModel."""

    def test_invalid_change_params_raises(self):
        """Any param not in prf_model.parameter_names raises ValueError."""
        with pytest.raises(ValueError, match="not_a_param"):
            CenterSurroundPRFModel(
                prf_model=Gaussian2DPRFResponse(),
                change_params=["not_a_param"],
            )

    def test_non_default_change_params_parameter_names(self):
        """change_params=['mu_x'] splits mu_x into mu_x_center and mu_x_surround."""
        model = CenterSurroundPRFModel(
            prf_model=Gaussian2DPRFResponse(),
            change_params=["mu_x"],
        )
        assert "mu_x_center" in model.parameter_names
        assert "mu_x_surround" in model.parameter_names
        assert "mu_x" not in model.parameter_names

    def test_multiple_change_params_parameter_names(self):
        """Multiple change_params each split into center/surround variants."""
        model = CenterSurroundPRFModel(
            prf_model=Gaussian2DPRFResponse(),
            change_params=["mu_x", "sigma"],
        )
        for param in ("mu_x", "sigma"):
            assert f"{param}_center" in model.parameter_names
            assert f"{param}_surround" in model.parameter_names
            assert param not in model.parameter_names


class TestInitDogFromGaussian:
    """Tests for init_dog_from_gaussian utility function."""

    @pytest.fixture
    def gaussian_params(self):
        """Dataframe with Gaussian model parameters."""
        return pd.DataFrame(
            {
                "mu_x": [1.0, -1.0],
                "mu_y": [0.5, -0.5],
                "sigma": [2.0, 3.0],
                "delay": [6.0, 6.0],
                "dispersion": [0.9, 0.9],
                "undershoot": [12.0, 12.0],
                "u_dispersion": [0.9, 0.9],
                "ratio": [0.48, 0.48],
                "weight_deriv": [-0.5, -0.5],
                "baseline": [5.0, 8.0],
                "amplitude": [10.0, 7.0],
            },
        )

    def test_output_columns(self, gaussian_params: pd.DataFrame):
        """Output DataFrame has the expected DoG column set."""
        dog_params = init_dog_from_gaussian(gaussian_params)
        expected_cols = {
            "mu_x",
            "mu_y",
            "sigma_center",
            "sigma_surround",
            "delay",
            "dispersion",
            "undershoot",
            "u_dispersion",
            "ratio",
            "weight_deriv",
            "baseline",
            "amplitude_center",
            "amplitude_surround",
        }
        assert set(dog_params.columns) == expected_cols

    def test_sigma_mapping(self, gaussian_params: pd.DataFrame):
        """sigma_center and sigma_surround are mapped correctly from sigma and sigma_ratio."""
        dog_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=5.0)
        assert list(dog_params["sigma_center"]) == [2.0, 3.0]
        assert list(dog_params["sigma_surround"]) == [10.0, 15.0]

    def test_amplitude_mapping(self, gaussian_params: pd.DataFrame):
        """Amplitude is mapped to amplitude_center and amplitude_surround defaults to 0."""
        dog_params = init_dog_from_gaussian(gaussian_params)
        assert list(dog_params["amplitude_center"]) == [10.0, 7.0]
        assert list(dog_params["amplitude_surround"]) == [0.0, 0.0]

    def test_passthrough_columns(self, gaussian_params: pd.DataFrame):
        """Non-sigma/amplitude columns pass through unchanged."""
        dog_params = init_dog_from_gaussian(gaussian_params)
        for col in ["mu_x", "mu_y", "baseline", "delay", "dispersion"]:
            assert list(dog_params[col]) == list(gaussian_params[col])

    def test_no_sigma_or_amplitude_columns(self, gaussian_params: pd.DataFrame):
        """Output DataFrame does not retain the original sigma or amplitude columns."""
        dog_params = init_dog_from_gaussian(gaussian_params)
        assert "sigma" not in dog_params.columns
        assert "amplitude" not in dog_params.columns

    def test_custom_sigma_ratio(self, gaussian_params: pd.DataFrame):
        """Custom sigma_ratio is applied correctly to compute sigma_surround."""
        dog_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0)
        np.testing.assert_allclose(dog_params["sigma_surround"].to_numpy(), gaussian_params["sigma"].to_numpy() * 3.0)

    def test_sigma_surround_direct(self, gaussian_params: pd.DataFrame):
        """Providing sigma_surround directly uses that value for all rows."""
        dog_params = init_dog_from_gaussian(gaussian_params, sigma_surround=8.0)
        assert list(dog_params["sigma_surround"]) == [8.0, 8.0]

    def test_sigma_surround_equals_sigma_center_allowed(self, gaussian_params: pd.DataFrame):
        """sigma_surround equal to the largest sigma_center is valid."""
        dog_params = init_dog_from_gaussian(gaussian_params, sigma_surround=3.0)
        assert list(dog_params["sigma_surround"]) == [3.0, 3.0]

    def test_sigma_surround_too_small_raises(self, gaussian_params: pd.DataFrame):
        """sigma_surround smaller than any sigma_center row raises ValueError."""
        with pytest.raises(ValueError, match="sigma_surround"):
            init_dog_from_gaussian(gaussian_params, sigma_surround=1.0)

    def test_sigma_surround_overrides_sigma_ratio(self, gaussian_params: pd.DataFrame):
        """sigma_surround takes priority over sigma_ratio when both are provided."""
        dog_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0, sigma_surround=10.0)
        assert list(dog_params["sigma_surround"]) == [10.0, 10.0]

    def test_default_uses_ratio_5(self, gaussian_params: pd.DataFrame):
        """Default call (no sigma args) applies sigma_ratio=5.0."""
        dog_params = init_dog_from_gaussian(gaussian_params)
        np.testing.assert_allclose(dog_params["sigma_surround"].to_numpy(), gaussian_params["sigma"].to_numpy() * 5.0)
