"""Tests for CSF model classes and functions."""

import numpy as np
import pandas as pd
import pytest
from keras import ops
from prfmodel.exceptions import ShapeError
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.fitters import SGDFitter
from prfmodel.models.csf import CSFModel
from prfmodel.models.csf import CSFTuning
from prfmodel.models.csf import predict_contrast_response
from prfmodel.models.csf import predict_contrast_sensitivity
from prfmodel.models.csf.canonical import CanonicalCSFModel
from prfmodel.stimuli import CSFStimulus
from .conftest import parametrize_dtype


class TestSetup:
    """Setup parameters and objects for testing."""

    num_voxels = 3
    num_frames = 8

    @pytest.fixture
    def sf(self) -> np.ndarray:
        """Spatial frequencies."""
        return np.array([0.5, 1.0, 3.0, 6.0, 12.0, 3.0, 1.0, 0.5])

    @pytest.fixture
    def contrast(self) -> np.ndarray:
        """Michelson contrast in percent."""
        return np.array([5.0, 10.0, 20.0, 40.0, 80.0, 40.0, 10.0, 5.0])

    @pytest.fixture
    def stimulus(self, sf: np.ndarray, contrast: np.ndarray) -> CSFStimulus:
        """CSF stimulus object."""
        return CSFStimulus(
            sf=sf,
            contrast=contrast,
            frame_duration=1.0,
        )


class TestPredictContrastSensitivity(TestSetup):
    """Tests for predict_contrast_sensitivity function."""

    @pytest.fixture
    def cs_peak(self) -> np.ndarray:
        """Peak contrast sensitivity for NUM_VOXELS voxels, shape (NUM_VOXELS, 1)."""
        return np.array([[100.0], [150.0], [80.0]])

    @pytest.fixture
    def sf_peak(self) -> np.ndarray:
        """Peak spatial frequency for NUM_VOXELS voxels, shape (NUM_VOXELS, 1)."""
        return np.array([[2.5], [3.0], [4.0]])

    @pytest.fixture
    def width_r(self) -> np.ndarray:
        """Right branch width for NUM_VOXELS voxels, shape (NUM_VOXELS, 1)."""
        return np.array([[1.28], [1.5], [1.0]])

    def test_shape_error_cs_peak(self, sf: np.ndarray, sf_peak: np.ndarray, width_r: np.ndarray) -> None:
        """Test that ShapeError is raised when cs_peak has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_sensitivity(sf, np.array([100.0]), sf_peak, 0.68, width_r)

    def test_shape_error_sf_peak(self, sf: np.ndarray, cs_peak: np.ndarray, width_r: np.ndarray) -> None:
        """Test that ShapeError is raised when sf_peak has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_sensitivity(sf, cs_peak, np.array([3.0]), 0.68, width_r)

    def test_shape_error_width_r(self, sf: np.ndarray, cs_peak: np.ndarray, sf_peak: np.ndarray) -> None:
        """Test that ShapeError is raised when width_r has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_sensitivity(sf, cs_peak, sf_peak, 0.68, np.array([1.28]))

    def test_shape_mismatch_error(self, sf: np.ndarray) -> None:
        """Test that ShapeMismatchError is raised when parameter batch dimensions are inconsistent."""
        cs_peak_2 = np.array([[100.0], [150.0]])  # (2, 1)
        sf_peak_3 = np.array([[2.5], [3.0], [4.0]])  # (3, 1)
        width_r_2 = np.array([[1.28], [1.5]])  # (2, 1)
        with pytest.raises(ShapeMismatchError):
            predict_contrast_sensitivity(sf, cs_peak_2, sf_peak_3, 0.68, width_r_2)

    @parametrize_dtype
    def test_output_shape(
        self,
        sf: np.ndarray,
        cs_peak: np.ndarray,
        sf_peak: np.ndarray,
        width_r: np.ndarray,
        dtype: str,
    ) -> None:
        """Test that output has correct shape."""
        result = predict_contrast_sensitivity(sf, cs_peak, sf_peak, 0.68, width_r, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)

    def test_peak_sf_equals_cs_peak(self) -> None:
        """Test that sensitivity equals cs_peak when sf equals sf_peak."""
        sf_peak_val, cs_peak_val = 3.0, 100.0
        sf = np.array([sf_peak_val])
        cs_peak = np.array([[cs_peak_val]])
        sf_peak = np.array([[sf_peak_val]])
        width_r = np.array([[1.28]])

        result = float(ops.convert_to_numpy(predict_contrast_sensitivity(sf, cs_peak, sf_peak, 0.68, width_r)).flat[0])

        assert np.allclose(result, cs_peak_val)

    def test_left_vs_right_branch(self, sf: np.ndarray) -> None:
        """Test that wider left branch yields lower sensitivity for sf below sf_peak."""
        sf_peak_val = 3.0
        cs_peak = np.array([[100.0]])
        sf_peak = np.array([[sf_peak_val]])
        width_r = np.array([[1.28]])

        result_wide_l = ops.convert_to_numpy(predict_contrast_sensitivity(sf, cs_peak, sf_peak, 10.0, width_r))
        result_narrow_l = ops.convert_to_numpy(predict_contrast_sensitivity(sf, cs_peak, sf_peak, 0.1, width_r))

        # Wider left branch drops sensitivity faster → lower values left of peak
        left_frames = sf < sf_peak_val
        assert np.all(result_wide_l[0, left_frames] < result_narrow_l[0, left_frames])


class TestPredictContrastResponse(TestSetup):
    """Tests for predict_contrast_response function."""

    @pytest.fixture
    def sensitivity(self) -> np.ndarray:
        """Contrast sensitivity."""
        return np.full((self.num_voxels, self.num_frames), 100.0)

    @pytest.fixture
    def slope_crf(self) -> np.ndarray:
        """Naka-Rushton slope."""
        return np.array([[2.0], [2.0], [3.0]])

    def test_shape_error_sensitivity(self, contrast: np.ndarray, slope_crf: np.ndarray) -> None:
        """Test that ShapeError is raised when sensitivity has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_response(contrast, np.ones(self.num_frames), slope_crf)

    def test_shape_error_slope_crf(self, contrast: np.ndarray, sensitivity: np.ndarray) -> None:
        """Test that ShapeError is raised when slope_crf has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_response(contrast, sensitivity, np.array([2.0]))

    def test_shape_mismatch_error(self, contrast: np.ndarray) -> None:
        """Test that ShapeMismatchError is raised when sensitivity and slope_crf batch dims differ."""
        sensitivity_2 = np.ones((2, self.num_frames))
        slope_crf_3 = np.array([[2.0], [2.0], [2.0]])
        with pytest.raises(ShapeMismatchError):
            predict_contrast_response(contrast, sensitivity_2, slope_crf_3)

    @parametrize_dtype
    def test_output_shape(
        self,
        contrast: np.ndarray,
        sensitivity: np.ndarray,
        slope_crf: np.ndarray,
        dtype: str,
    ) -> None:
        """Test that output has correct shape."""
        result = predict_contrast_response(contrast, sensitivity, slope_crf, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)

    def test_output_range(
        self,
        contrast: np.ndarray,
        sensitivity: np.ndarray,
        slope_crf: np.ndarray,
    ) -> None:
        """Test that output values are in [0, 1] as expected from the Naka-Rushton CRF."""
        result = ops.convert_to_numpy(predict_contrast_response(contrast, sensitivity, slope_crf))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_known_naka_rushton_value(self) -> None:
        """Test Naka-Rushton formula against the analytical expression at a single operating point.

        At contrast c (in percent), sensitivity s, and slope q: R = c^q / (c^q + (100/s)^q).

        """
        c, s, q = 0.5, 100.0, 2.0
        contrast = np.array([c])
        sensitivity = np.array([[s]])
        slope_crf = np.array([[q]])

        result = float(ops.convert_to_numpy(predict_contrast_response(contrast, sensitivity, slope_crf)).flat[0])
        expected = c**q / (c**q + (100.0 / s) ** q)

        assert np.allclose(result, expected)

    def test_no_overflow_at_high_contrast_and_slope(self) -> None:
        """Test that a high contrast and steep slope saturate the CRF instead of overflowing to NaN.

        In float32, ``C^q`` overflows to ``inf`` for ``C = 100`` and ``q = 20`` (``1e40``), so evaluating
        ``C^q / (C^q + Q^q)`` directly yields ``inf / inf = NaN``.

        """
        contrast = np.array([100.0, 1.0])
        sensitivity = np.array([[100.0, 1e-3]])
        slope_crf = np.array([[20.0]])

        result = ops.convert_to_numpy(predict_contrast_response(contrast, sensitivity, slope_crf, dtype="float32"))

        assert np.all(np.isfinite(result))
        assert np.allclose(result, [[1.0, 0.0]])

    @pytest.mark.float64
    def test_zero_contrast_gives_zero_response(self) -> None:
        """Test that a blank frame, with zero contrast, responds with exactly zero even for a shallow slope.

        Raising zero contrast to a floor of ``eps = 1e-10`` instead gives ``eps^q / (eps^q + Q^q) = 0.09`` here.

        """
        contrast = np.array([0.0, 10.0])
        sensitivity = np.array([[100.0, 100.0]])
        slope_crf = np.array([[0.1]])

        result = ops.convert_to_numpy(predict_contrast_response(contrast, sensitivity, slope_crf, dtype="float64"))

        assert result[0, 0] == 0.0
        assert result[0, 1] > 0.0


class TestCSFTuning(TestSetup):
    """Tests for CSFTuning class."""

    @pytest.fixture
    def params(self) -> pd.DataFrame:
        """CSF parameters."""
        return pd.DataFrame(
            {
                "cs_peak": [100.0, 150.0, 80.0],
                "sf_peak": [2.5, 3.0, 4.0],
                "width_r": [1.28, 1.5, 1.0],
                "slope_crf": [2.0, 2.0, 3.0],
            },
        )

    def test_parameter_names(self) -> None:
        """Test that parameter names are correct."""
        model = CSFTuning()
        assert model.parameter_names == ["cs_peak", "sf_peak", "width_r", "slope_crf"]

    @parametrize_dtype
    def test_predict(self, stimulus: CSFStimulus, params: pd.DataFrame, dtype: str) -> None:
        """Test that response prediction returns correct shape."""
        model = CSFTuning()
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)

    @pytest.mark.float64
    def test_matches_composed_functions(self, stimulus: CSFStimulus, params: pd.DataFrame) -> None:
        """Test that the log-domain tuning response equals composing the public CSF and CRF functions."""
        model = CSFTuning()
        result = model(stimulus, params, dtype="float64")

        sensitivity = predict_contrast_sensitivity(
            stimulus.sf,
            params[["cs_peak"]].to_numpy(),
            params[["sf_peak"]].to_numpy(),
            model.width_l,
            params[["width_r"]].to_numpy(),
            dtype="float64",
        )
        expected = predict_contrast_response(
            stimulus.contrast,
            sensitivity,
            params[["slope_crf"]].to_numpy(),
            dtype="float64",
        )

        assert np.allclose(result, ops.convert_to_numpy(expected))

    def test_extreme_parameters_fit_to_finite_parameters(self) -> None:
        """Test that extreme CSF parameters do not poison the gradient in float32.

        A low peak sensitivity with a steep right branch drives the sensitivity far below one at high spatial
        frequencies, and a steep CRF slope at high contrast makes the CRF powers overflow. Forming ``10^x``,
        ``C^q``, and ``Q^q`` in float32 then underflows or overflows and yields NaN gradients, even where the
        forward value is fine. Only a gradient reveals it, so this fits rather than merely predicting.

        """
        stimulus = CSFStimulus(
            sf=np.array([0.5, 1.0, 3.0, 6.0, 12.0, 18.0, 1.0, 0.5]),
            # Percent contrast, including blank frames
            contrast=np.array([0.0, 0.25, 1.0, 5.0, 20.0, 80.0, 0.0, 100.0]),
            frame_duration=1.0,
        )
        model = CSFModel(impulse_model=None)

        true_params = pd.DataFrame(
            {
                "cs_peak": [100.0, 100.0],
                "sf_peak": [2.5, 2.5],
                "width_r": [1.28, 1.28],
                "slope_crf": [2.0, 2.0],
                "baseline": [0.0, 0.0],
                "amplitude": [1.0, 1.0],
            },
        )
        init_params = pd.DataFrame(
            {
                "cs_peak": [2.0, 100.0],
                "sf_peak": [1.0, 3.0],
                "width_r": [3.0, 1.28],
                "slope_crf": [8.0, 20.0],
                "baseline": [0.0, 0.0],
                "amplitude": [1.0, 1.0],
            },
        )

        fitter = SGDFitter(model=model, stimulus=stimulus, dtype="float32")
        _, fitted = fitter.fit(model(stimulus, true_params, dtype="float32"), init_params, num_steps=5)

        assert np.all(np.isfinite(fitted.to_numpy())), f"Fit produced non-finite parameters: {fitted}"

    def test_blank_frames_with_zero_spatial_frequency(self, params: pd.DataFrame) -> None:
        """Test that blank frames stored as recorded, with zero spatial frequency, respond with zero and fit.

        The log of a zero spatial frequency is ``-inf``, which leaves the forward value at zero but makes the gradients
        NaN unless the model keeps blank frames out of the log. A grating at zero spatial frequency also responds with
        zero, the limit of the response as the spatial frequency goes to zero.

        """
        stimulus = CSFStimulus(
            sf=np.array([0.0, 1.0, 3.0, 0.0, 6.0, 0.0]),
            contrast=np.array([0.0, 20.0, 80.0, 0.0, 5.0, 50.0]),
            frame_duration=1.0,
        )
        is_blank = stimulus.sf == 0

        response = ops.convert_to_numpy(CSFTuning()(stimulus, params, dtype="float32"))

        np.testing.assert_array_equal(response[:, is_blank], 0.0)
        assert np.all(response[:, ~is_blank] > 0.0)

        model = CSFModel(impulse_model=None)
        full_params = params.assign(baseline=0.0, amplitude=1.0)
        init_params = full_params.assign(cs_peak=50.0, slope_crf=4.0)

        fitter = SGDFitter(model=model, stimulus=stimulus, dtype="float32")
        _, fitted = fitter.fit(model(stimulus, full_params, dtype="float32"), init_params, num_steps=5)

        assert np.all(np.isfinite(fitted.to_numpy())), f"Fit produced non-finite parameters: {fitted}"


class TestCSFModel(TestSetup):
    """Tests for CSFModel and SimpleCSFModel convenience wrappers."""

    @pytest.fixture
    def params(self) -> pd.DataFrame:
        """Full parameter set for CSFModel with NUM_VOXELS voxels."""
        return pd.DataFrame(
            {
                "cs_peak": [100.0, 150.0, 80.0],
                "sf_peak": [2.5, 3.0, 4.0],
                "width_r": [1.28, 1.5, 1.0],
                "slope_crf": [2.0, 2.0, 3.0],
                "baseline": [0.0, 0.0, 0.0],
                "amplitude": [1.0, 1.0, 1.0],
                "delay": [6.0, 6.0, 6.0],
                "dispersion": [1.0, 1.0, 1.0],
                "undershoot": [16.0, 16.0, 16.0],
                "u_dispersion": [1.0, 1.0, 1.0],
                "ratio": [0.1667, 0.1667, 0.1667],
                "weight_deriv": [0.0, 0.0, 0.0],
            },
        )

    def test_parameter_names(self) -> None:
        """Test that CSFModel exposes all expected parameter names."""
        model = CSFModel()
        params = model.parameter_names
        assert "cs_peak" in params
        assert "sf_peak" in params
        assert "width_r" in params
        assert "slope_crf" in params
        assert "baseline" in params
        assert "amplitude" in params

    @parametrize_dtype
    def test_predict(self, stimulus: CSFStimulus, params: pd.DataFrame, dtype: str) -> None:
        """Test that CSFModel prediction returns correct shape."""
        model = CSFModel()
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)

    @parametrize_dtype
    def test_predict_no_hrf(self, stimulus: CSFStimulus, params: pd.DataFrame, dtype: str) -> None:
        """Test that prediction without HRF and temporal model returns correct shape."""
        model = CSFModel(impulse_model=None, scaling_model=None)
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)

    @parametrize_dtype
    def test_canonical_csf_model_predict(self, stimulus: CSFStimulus, params: pd.DataFrame, dtype: str) -> None:
        """Test that SimpleCSFModel with CSFTuning returns correct shape."""
        model = CanonicalCSFModel(csf_model=CSFTuning())
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)
