"""Tests for divisive normalization models and initialization functions."""

import warnings
import numpy as np
import pandas as pd
import pytest
from keras import ops
from pytest_regressions.num_regression import NumericRegressionFixture
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.prf import DivNormGaussian2DPRFModel
from prfmodel.models.prf import Gaussian2DPRFResponse
from prfmodel.models.prf import init_div_norm_from_dog_css
from prfmodel.models.prf._stimulus_encoding import PRFStimulusEncoder
from prfmodel.models.prf.canonical import DivNormPRFModel
from prfmodel.scaling import Baseline
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup


class TestDivNormGaussian2DPRFModel(PRFStimulusSetup):
    """Tests for the DivNormGaussian2DPRFModel class."""

    @pytest.fixture
    def prf_model(self):
        """PRF model object."""
        return DivNormGaussian2DPRFModel()

    @pytest.fixture
    def impulse_model(self):
        """Impulse model object."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def temporal_model(self):
        """Temporal model object."""
        return Baseline()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma_activation": [1.0, 1.5, 2.0],
                "sigma_normalization": [2.0, 3.0, 4.0],
                "weight_deriv": [-0.5, -0.5, -0.5],
                "amplitude_activation": [1.1, 1.0, 0.9],
                "baseline_activation": [0.0, 0.1, 0.5],
                "amplitude_normalization": [10.0, 5.0, 20.0],
                "baseline_normalization": [20.0, 10.0, 5.0],
                "baseline": [-0.5, 0.5, 2.0],
            },
        )

    def test_parameter_names(
        self,
        prf_model: DivNormGaussian2DPRFModel,
        impulse_model: DerivativeTwoGammaImpulse,
        temporal_model: Baseline,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        expected = [
            "mu_y",
            "mu_x",
            "sigma_activation",
            "sigma_normalization",
            "amplitude_activation",
            "amplitude_normalization",
            "baseline_activation",
            "baseline_normalization",
        ]
        expected.extend(impulse_model.parameter_names)
        expected.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(dict.fromkeys(expected))

    @pytest.mark.parametrize(
        ("impulse_model", "temporal_model"),
        [
            (DerivativeTwoGammaImpulse(), Baseline()),
            (DerivativeTwoGammaImpulse, Baseline),
            (DerivativeTwoGammaImpulse(), None),
            (None, Baseline()),
            (DerivativeTwoGammaImpulse, None),
            (None, Baseline),
            (None, None),
        ],
    )
    def test_predict(
        self,
        impulse_model: BaseImpulse,
        temporal_model: BaseScaling,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape."""
        prf_model = DivNormGaussian2DPRFModel(
            impulse_model=impulse_model,
            scaling_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    def test_activation_normalization_suffixes(self, prf_model: DivNormGaussian2DPRFModel):
        """Test that sigma uses activation/normalization suffixes, not center/surround."""
        assert "sigma_activation" in prf_model.parameter_names
        assert "sigma_normalization" in prf_model.parameter_names
        assert "sigma_center" not in prf_model.parameter_names
        assert "sigma_surround" not in prf_model.parameter_names

    @pytest.mark.parametrize("temporal_model", [None, Baseline()])
    @pytest.mark.parametrize("impulse_model", [None, DerivativeTwoGammaImpulse()])
    def test_predict_regression(
        self,
        num_regression: NumericRegressionFixture,
        impulse_model: BaseImpulse,
        temporal_model: BaseScaling,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference file."""
        prf_model = DivNormGaussian2DPRFModel(
            impulse_model=impulse_model,
            scaling_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4, "rtol": 1e-3},
        )


class TestInitDivNormParameters:
    """Tests for init_div_norm_from_dog_css utility function."""

    @pytest.fixture
    def dog_params(self):
        """Dataframe with Difference of Gaussian (DoG) model parameters."""
        return pd.DataFrame(
            {
                "mu_x": [1.0, -1.0],
                "mu_y": [0.5, -0.5],
                "sigma_center": [2.0, 3.0],
                "sigma_surround": [4.0, 9.0],
                "amplitude_center": [10.0, 7.0],
                "amplitude_surround": [5.0, 1.4],
                "delay": [6.0, 6.0],
                "dispersion": [0.9, 0.9],
                "undershoot": [12.0, 12.0],
                "u_dispersion": [0.9, 0.9],
                "ratio": [0.48, 0.48],
                "weight_deriv": [-0.5, -0.5],
                "baseline": [5.0, 8.0],
            },
        )

    def test_output_columns(self, dog_params: pd.DataFrame):
        """Output DataFrame has the expected DN column set; DoG sigma and amplitude columns are dropped."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        expected_cols = {
            "mu_x",
            "mu_y",
            "delay",
            "dispersion",
            "undershoot",
            "u_dispersion",
            "ratio",
            "weight_deriv",
            "baseline",
            "sigma_activation",
            "sigma_normalization",
            "amplitude_activation",
            "amplitude_normalization",
            "baseline_activation",
            "baseline_normalization",
        }
        assert set(dn_params.columns) == expected_cols

    def test_sigma_activation_equals_sigma_center(self, dog_params: pd.DataFrame):
        """sigma_activation equals the DoG sigma_center."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        assert list(dn_params["sigma_activation"]) == [2.0, 3.0]

    def test_sigma_normalization_equals_sigma_surround(self, dog_params: pd.DataFrame):
        """sigma_normalization equals the DoG sigma_surround."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        assert list(dn_params["sigma_normalization"]) == [4.0, 9.0]

    def test_amplitude_activation_constant_one(self, dog_params: pd.DataFrame):
        """amplitude_activation is set to 1.0 for all rows."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        assert list(dn_params["amplitude_activation"]) == [1.0, 1.0]

    def test_amplitude_normalization_constant_one(self, dog_params: pd.DataFrame):
        """amplitude_normalization is set to 1.0 for all rows."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        assert list(dn_params["amplitude_normalization"]) == [1.0, 1.0]

    def test_baseline_activation_from_dog_amplitude_ratio(self, dog_params: pd.DataFrame):
        """baseline_activation equals (amplitude_surround / amplitude_center) * baseline_normalization."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        # dog_amplitude_ratio = [5/10, 1.4/7] = [0.5, 0.2]; baseline_normalization = 50.0
        np.testing.assert_allclose(dn_params["baseline_activation"].to_numpy(), [25.0, 10.0])

    def test_baseline_normalization_from_css_n(self, dog_params: pd.DataFrame):
        """baseline_normalization equals css_n * 100 (default 0.5 -> 50.0)."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        assert list(dn_params["baseline_normalization"]) == [50.0, 50.0]

    def test_custom_css_n(self, dog_params: pd.DataFrame):
        """Custom css_n scales baseline_normalization by 100 and propagates to baseline_activation."""
        dn_params = init_div_norm_from_dog_css(dog_params, css_n=0.8)
        np.testing.assert_allclose(dn_params["baseline_normalization"].to_numpy(), 80.0)
        # baseline_activation = dog_amplitude_ratio * baseline_normalization = [0.5, 0.2] * 80
        np.testing.assert_allclose(dn_params["baseline_activation"].to_numpy(), [40.0, 16.0])

    @pytest.fixture
    def bar_stimulus(self):
        """2D bar pRF stimulus object."""
        return load_2d_prf_bar_stimulus()

    @staticmethod
    def _peak_normalization_drive(dn_params: pd.DataFrame, stimulus: PRFStimulus) -> np.ndarray:
        """Independently encode the normalization Gaussian to get the per-row peak drive."""
        norm_params = dn_params[["mu_x", "mu_y"]].copy()
        norm_params["sigma"] = dn_params["sigma_normalization"]
        response = Gaussian2DPRFResponse()(stimulus, norm_params)
        drive = PRFStimulusEncoder()(stimulus, response, norm_params)
        return np.asarray(ops.convert_to_numpy(ops.max(drive, axis=1)), dtype=np.float64)

    def test_baseline_normalization_uses_stimulus_drive(
        self,
        dog_params: pd.DataFrame,
        bar_stimulus: PRFStimulus,
    ):
        """With a stimulus, baseline_normalization equals peak_drive * css_n / (1 - css_n) per row."""
        css_n = 0.4
        dn_params = init_div_norm_from_dog_css(dog_params, css_n=css_n, stimulus=bar_stimulus)
        peak_drive = self._peak_normalization_drive(dn_params, bar_stimulus)
        expected = peak_drive * css_n / (1.0 - css_n)
        np.testing.assert_allclose(dn_params["baseline_normalization"].to_numpy(), expected, rtol=1e-5)

    def test_stimulus_changes_baseline_normalization(
        self,
        dog_params: pd.DataFrame,
        bar_stimulus: PRFStimulus,
    ):
        """The stimulus-aware mapping differs from the css_n * 100 fallback."""
        with_stim = init_div_norm_from_dog_css(dog_params, css_n=0.4, stimulus=bar_stimulus)
        without_stim = init_div_norm_from_dog_css(dog_params, css_n=0.4)
        assert not np.allclose(
            with_stim["baseline_normalization"].to_numpy(),
            without_stim["baseline_normalization"].to_numpy(),
        )

    def test_css_n_geq_1_with_stimulus_warns(
        self,
        dog_params: pd.DataFrame,
        bar_stimulus: PRFStimulus,
    ):
        """css_n >= 1.0 with a stimulus emits a UserWarning (non-positive baseline_normalization)."""
        with pytest.warns(UserWarning, match="css_n"):
            init_div_norm_from_dog_css(dog_params, css_n=1.0, stimulus=bar_stimulus)

    def test_passthrough_columns(self, dog_params: pd.DataFrame):
        """Columns unrelated to divisive normalization pass through unchanged."""
        dn_params = init_div_norm_from_dog_css(dog_params)
        for col in ["mu_x", "mu_y", "delay", "dispersion", "weight_deriv", "baseline"]:
            assert list(dn_params[col]) == list(dog_params[col])

    def test_input_not_mutated(self, dog_params: pd.DataFrame):
        """The input DataFrame is copied, not modified in place."""
        original = dog_params.copy()
        init_div_norm_from_dog_css(dog_params)
        pd.testing.assert_frame_equal(dog_params, original)

    def test_dog_sigma_ratio_below_1_warns(self, dog_params: pd.DataFrame):
        """A sigma_surround / sigma_center ratio below 1.0 emits a UserWarning."""
        dog_params = dog_params.copy()
        dog_params["sigma_surround"] = [1.0, 2.0]
        with pytest.warns(UserWarning, match="dog_sigma_ratio"):
            init_div_norm_from_dog_css(dog_params)

    def test_dog_amplitude_ratio_negative_warns(self, dog_params: pd.DataFrame):
        """A negative amplitude_surround / amplitude_center ratio emits a UserWarning."""
        dog_params = dog_params.copy()
        dog_params["amplitude_surround"] = [-5.0, -1.4]
        with pytest.warns(UserWarning, match="dog_amplitude_ratio"):
            init_div_norm_from_dog_css(dog_params)

    def test_css_n_negative_warns(self, dog_params: pd.DataFrame):
        """Negative css_n emits a UserWarning."""
        with pytest.warns(UserWarning, match="css_n"):
            init_div_norm_from_dog_css(dog_params, css_n=-0.1)

    def test_no_warning_for_valid_defaults(self, dog_params: pd.DataFrame):
        """Default arguments do not emit any warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            init_div_norm_from_dog_css(dog_params)

    def test_css_n_wrong_length_raises(self, dog_params: pd.DataFrame):
        """css_n longer than 1 but not matching the row count raises ValueError."""
        with pytest.raises(ValueError, match="css_n"):
            init_div_norm_from_dog_css(dog_params, css_n=np.array([0.5, 0.6, 0.7]))

    def test_per_voxel_css_n_allowed(self, dog_params: pd.DataFrame):
        """An array-like css_n whose length equals the row count is applied per voxel."""
        dn_params = init_div_norm_from_dog_css(dog_params, css_n=np.array([0.5, 0.6]))
        np.testing.assert_allclose(dn_params["baseline_normalization"].to_numpy(), [50.0, 60.0])
        # baseline_activation = dog_amplitude_ratio * baseline_normalization, per row
        np.testing.assert_allclose(dn_params["baseline_activation"].to_numpy(), [0.5 * 50.0, 0.2 * 60.0])

    def test_length_one_css_n_array_allowed(self, dog_params: pd.DataFrame):
        """A length-1 css_n array is treated as a single value and broadcasts across rows."""
        dn_params = init_div_norm_from_dog_css(dog_params, css_n=np.array([0.5]))
        np.testing.assert_allclose(dn_params["baseline_normalization"].to_numpy(), 50.0)


class TestDivNormPRFModel(PRFStimulusSetup):
    """Tests for the general DivNormPRFModel with a single shared pRF model."""

    @pytest.fixture
    def params(self):
        """Parameters for two Gaussian pRF models sharing mu_x, mu_y."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0],
                "mu_y": [1.0, 0.0],
                "sigma_activation": [1.0, 1.5],
                "sigma_normalization": [2.0, 3.0],
                "delay": [6.0, 7.0],
                "dispersion": [0.9, 1.0],
                "undershoot": [12.0, 11.0],
                "u_dispersion": [0.9, 1.0],
                "ratio": [0.48, 0.48],
                "weight_deriv": [0.5, 0.5],
                "amplitude_activation": [1.1, 1.0],
                "baseline_activation": [0.0, 0.1],
                "amplitude_normalization": [1.0, 1.0],
                "baseline_normalization": [1.0, 1.0],
                "baseline": [-0.5, 0.5],
            },
        )

    def test_parameter_names_two_gaussians(self):
        """A Gaussian model with shared mu_x/mu_y produces activation/normalization sigma suffixes."""
        model = DivNormPRFModel(
            prf_model=Gaussian2DPRFResponse(),
            shared_params=["mu_x", "mu_y"],
            impulse_model=None,
            scaling_model=None,
        )
        names = model.parameter_names
        assert "mu_x" in names
        assert "mu_y" in names
        assert "sigma_activation" in names
        assert "sigma_normalization" in names
        # Shared params appear exactly once (no suffix)
        assert "mu_x_activation" not in names
        assert "mu_x_normalization" not in names
        assert names.count("mu_x") == 1
        assert names.count("mu_y") == 1

    def test_parameter_names_no_shared(self):
        """With no shared params all pRF params get suffixes."""
        model = DivNormPRFModel(
            prf_model=Gaussian2DPRFResponse(),
            shared_params=[],
            impulse_model=None,
            scaling_model=None,
        )
        names = model.parameter_names
        assert "mu_y_activation" in names
        assert "mu_x_activation" in names
        assert "sigma_activation" in names
        assert "mu_y_normalization" in names
        assert "mu_x_normalization" in names
        assert "sigma_normalization" in names
        assert "mu_x" not in names
        assert "mu_y" not in names

    def test_invalid_shared_param_raises(self):
        """Providing a shared_param not in the pRF model raises ValueError."""
        with pytest.raises(ValueError, match="Shared parameters"):
            DivNormPRFModel(
                prf_model=Gaussian2DPRFResponse(),
                shared_params=["nonexistent"],
            )

    def test_predict_shape(self, stimulus: PRFStimulus, params: pd.DataFrame):
        """Model prediction has shape (num_voxels, num_frames)."""
        model = DivNormPRFModel(
            prf_model=Gaussian2DPRFResponse(),
            shared_params=["mu_x", "mu_y"],
        )
        resp = model(stimulus, params)
        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    def test_matches_gaussian2d_subclass(self, stimulus: PRFStimulus, params: pd.DataFrame):
        """DivNormPRFModel with a shared Gaussian and mu_x/mu_y matches DivNormGaussian2DPRFModel output."""
        general = DivNormPRFModel(
            prf_model=Gaussian2DPRFResponse(),
            shared_params=["mu_x", "mu_y"],
        )
        specific = DivNormGaussian2DPRFModel()
        np.testing.assert_allclose(
            np.array(general(stimulus, params)),
            np.array(specific(stimulus, params)),
            rtol=1e-5,
        )
