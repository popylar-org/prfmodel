"""Tests for the DivNormPRFModel and init_dn_from_gaussian."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.base import BaseImpulse
from prfmodel.models.base import BaseTemporal
from prfmodel.models.divisive_normalization import DivNormPRFModel
from prfmodel.models.divisive_normalization import init_dn_from_gaussian
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.temporal import DivNormAmplitude
from prfmodel.stimuli.prf import PRFStimulus
from tests.conftest import PRFStimulusSetup


class TestDivNormPRFModel(PRFStimulusSetup):
    """Tests for the DivNormPRFModel class."""

    @pytest.fixture
    def prf_model(self):
        """PRF model object."""
        return DivNormPRFModel()

    @pytest.fixture
    def impulse_model(self):
        """Impulse response model object."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def temporal_model(self):
        """Temporal model object."""
        return DivNormAmplitude()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma_activation": [1.0, 1.5, 2.0],
                "sigma_normalization": [2.0, 3.0, 4.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "amplitude_activation": [1.1, 1.0, 0.9],
                "baseline_activation": [0.0, 0.1, 0.5],
                "amplitude_normalization": [1.0, 1.0, 1.0],
                "baseline_normalization": [1.0, 1.0, 1.0],
            },
        )

    def test_parameter_names(
        self,
        prf_model: DivNormPRFModel,
        impulse_model: DerivativeTwoGammaImpulse,
        temporal_model: DivNormAmplitude,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        expected = ["mu_y", "mu_x", "sigma_activation", "sigma_normalization"]
        expected.extend(impulse_model.parameter_names)
        expected.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(dict.fromkeys(expected))

    @pytest.mark.parametrize(
        ("impulse_model", "temporal_model"),
        [
            (DerivativeTwoGammaImpulse(), DivNormAmplitude()),
            (DerivativeTwoGammaImpulse, DivNormAmplitude),
            (DerivativeTwoGammaImpulse(), None),
            (None, DivNormAmplitude()),
            (DerivativeTwoGammaImpulse, None),
            (None, DivNormAmplitude),
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
        prf_model = DivNormPRFModel(
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    def test_predict_responses(
        self,
        prf_model: DivNormPRFModel,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that predict_responses returns stacked tensor with correct shape."""
        resp = prf_model.predict_responses(stimulus, params)

        assert resp.shape == (params.shape[0], 2, stimulus.design.shape[0])

    def test_activation_normalization_suffixes(self, prf_model: DivNormPRFModel):
        """Test that sigma uses activation/normalization suffixes, not center/surround."""
        assert "sigma_activation" in prf_model.parameter_names
        assert "sigma_normalization" in prf_model.parameter_names
        assert "sigma_center" not in prf_model.parameter_names
        assert "sigma_surround" not in prf_model.parameter_names


class TestInitDnFromGaussian:
    """Tests for init_dn_from_gaussian utility function."""

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
        """Output DataFrame has the expected DN column set; baseline is dropped."""
        dn_params = init_dn_from_gaussian(gaussian_params)
        expected_cols = {
            "mu_x",
            "mu_y",
            "sigma_activation",
            "sigma_normalization",
            "delay",
            "dispersion",
            "undershoot",
            "u_dispersion",
            "ratio",
            "weight_deriv",
            "amplitude_activation",
            "baseline_activation",
            "amplitude_normalization",
            "baseline_normalization",
        }
        assert set(dn_params.columns) == expected_cols

    def test_no_sigma_or_amplitude_columns(self, gaussian_params: pd.DataFrame):
        """Output DataFrame does not retain the original sigma or amplitude columns."""
        dn_params = init_dn_from_gaussian(gaussian_params)
        assert "sigma" not in dn_params.columns
        assert "amplitude" not in dn_params.columns

    def test_sigma_mapping(self, gaussian_params: pd.DataFrame):
        """sigma_activation equals sigma; sigma_normalization equals sigma * sigma_ratio."""
        dn_params = init_dn_from_gaussian(gaussian_params, sigma_ratio=2.0)
        assert list(dn_params["sigma_activation"]) == [2.0, 3.0]
        assert list(dn_params["sigma_normalization"]) == [4.0, 6.0]

    def test_amplitude_mapping(self, gaussian_params: pd.DataFrame):
        """amplitude_activation equals amplitude from Gaussian params."""
        dn_params = init_dn_from_gaussian(gaussian_params)
        assert list(dn_params["amplitude_activation"]) == [10.0, 7.0]

    def test_baseline_mapped_from_gaussian(self, gaussian_params: pd.DataFrame):
        """baseline_activation is read from the gaussian_params baseline column by default."""
        dn_params = init_dn_from_gaussian(gaussian_params)
        assert list(dn_params["baseline_activation"]) == [5.0, 8.0]
        assert list(dn_params["amplitude_normalization"]) == [1.0, 1.0]
        assert list(dn_params["baseline_normalization"]) == [1.0, 1.0]

    def test_baseline_activation_missing_raises(self, gaussian_params: pd.DataFrame):
        """ValueError when baseline_activation=None and no baseline column present."""
        params_no_baseline = gaussian_params.drop(columns=["baseline"])
        with pytest.raises(ValueError, match="baseline_activation"):
            init_dn_from_gaussian(params_no_baseline)

    def test_custom_baseline_activation(self, gaussian_params: pd.DataFrame):
        """Custom baseline_activation (b) is applied to all rows."""
        dn_params = init_dn_from_gaussian(gaussian_params, baseline_activation=0.5)
        assert list(dn_params["baseline_activation"]) == [0.5, 0.5]

    def test_custom_amplitude_normalization(self, gaussian_params: pd.DataFrame):
        """Custom amplitude_normalization (c) is applied to all rows."""
        dn_params = init_dn_from_gaussian(gaussian_params, amplitude_normalization=2.0)
        assert list(dn_params["amplitude_normalization"]) == [2.0, 2.0]

    def test_custom_baseline_normalization(self, gaussian_params: pd.DataFrame):
        """Custom baseline_normalization (d) is applied to all rows."""
        dn_params = init_dn_from_gaussian(gaussian_params, baseline_normalization=0.5)
        assert list(dn_params["baseline_normalization"]) == [0.5, 0.5]

    def test_passthrough_columns(self, gaussian_params: pd.DataFrame):
        """Non-sigma/amplitude/baseline columns pass through unchanged."""
        dn_params = init_dn_from_gaussian(gaussian_params)
        for col in ["mu_x", "mu_y", "delay", "dispersion"]:
            assert list(dn_params[col]) == list(gaussian_params[col])

    def test_custom_sigma_ratio(self, gaussian_params: pd.DataFrame):
        """Custom sigma_ratio is applied correctly to compute sigma_normalization."""
        dn_params = init_dn_from_gaussian(gaussian_params, sigma_ratio=3.0)
        np.testing.assert_allclose(
            dn_params["sigma_normalization"].to_numpy(),
            gaussian_params["sigma"].to_numpy() * 3.0,
        )

    def test_default_uses_ratio_2(self, gaussian_params: pd.DataFrame):
        """Default call applies sigma_ratio=2.0."""
        dn_params = init_dn_from_gaussian(gaussian_params)
        np.testing.assert_allclose(
            dn_params["sigma_normalization"].to_numpy(),
            gaussian_params["sigma"].to_numpy() * 2.0,
        )

    def test_sigma_normalization_direct(self, gaussian_params: pd.DataFrame):
        """Providing sigma_normalization directly uses that value for all rows."""
        dn_params = init_dn_from_gaussian(gaussian_params, sigma_normalization=8.0)
        assert list(dn_params["sigma_normalization"]) == [8.0, 8.0]

    def test_sigma_normalization_equals_sigma_activation_allowed(self, gaussian_params: pd.DataFrame):
        """sigma_normalization equal to the largest sigma_activation is valid (phi=1)."""
        dn_params = init_dn_from_gaussian(gaussian_params, sigma_normalization=3.0)
        assert list(dn_params["sigma_normalization"]) == [3.0, 3.0]

    def test_sigma_normalization_too_small_raises(self, gaussian_params: pd.DataFrame):
        """sigma_normalization smaller than any sigma_activation row raises ValueError."""
        with pytest.raises(ValueError, match="sigma_normalization"):
            init_dn_from_gaussian(gaussian_params, sigma_normalization=1.0)

    def test_sigma_normalization_overrides_sigma_ratio(self, gaussian_params: pd.DataFrame):
        """sigma_normalization takes priority over sigma_ratio when both are provided."""
        dn_params = init_dn_from_gaussian(gaussian_params, sigma_ratio=3.0, sigma_normalization=10.0)
        assert list(dn_params["sigma_normalization"]) == [10.0, 10.0]

    def test_sigma_ratio_less_than_1_raises(self, gaussian_params: pd.DataFrame):
        """sigma_ratio < 1.0 raises ValueError (phi must be >= 1)."""
        with pytest.raises(ValueError, match="sigma_ratio"):
            init_dn_from_gaussian(gaussian_params, sigma_ratio=0.5)

    def test_sigma_ratio_exactly_1_allowed(self, gaussian_params: pd.DataFrame):
        """sigma_ratio=1.0 is valid (phi=1 means equal-sized pRFs)."""
        dn_params = init_dn_from_gaussian(gaussian_params, sigma_ratio=1.0)
        np.testing.assert_allclose(
            dn_params["sigma_normalization"].to_numpy(),
            dn_params["sigma_activation"].to_numpy(),
        )
