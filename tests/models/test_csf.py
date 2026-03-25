"""Tests for CSF model classes and functions."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.base import BatchDimensionError
from prfmodel.models.base import ShapeError
from prfmodel.models.composite import SimpleCSFModel
from prfmodel.models.csf import CSFModel
from prfmodel.models.csf import CSFResponse
from prfmodel.models.csf import predict_contrast_response
from prfmodel.models.csf import predict_contrast_sensitivity
from prfmodel.stimuli.csf import CSFStimulus
from .conftest import parametrize_dtype


class TestSetup:
    """Setup parameters and objects for testing."""

    num_voxels = 3
    num_frames = 8

    @pytest.fixture
    def sf(self) -> np.ndarray:
        """Spatial frequencies for NUM_FRAMES frames."""
        return np.array([0.5, 1.0, 3.0, 6.0, 12.0, 3.0, 1.0, 0.5])

    @pytest.fixture
    def contrast(self) -> np.ndarray:
        """Michelson contrast for NUM_FRAMES frames."""
        return np.array([0.05, 0.1, 0.2, 0.4, 0.8, 0.4, 0.1, 0.05])

    @pytest.fixture
    def stimulus(self, sf: np.ndarray, contrast: np.ndarray) -> CSFStimulus:
        """CSF stimulus object."""
        return CSFStimulus(
            sf=sf,
            contrast=contrast,
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

    def test_batch_dimension_error(self, sf: np.ndarray) -> None:
        """Test that BatchDimensionError is raised when parameter batch dimensions are inconsistent."""
        cs_peak_2 = np.array([[100.0], [150.0]])  # (2, 1)
        sf_peak_3 = np.array([[2.5], [3.0], [4.0]])  # (3, 1)
        width_r_2 = np.array([[1.28], [1.5]])  # (2, 1)
        with pytest.raises(BatchDimensionError):
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

        result = float(np.asarray(predict_contrast_sensitivity(sf, cs_peak, sf_peak, 0.68, width_r)).flat[0])

        assert np.allclose(result, cs_peak_val)

    def test_left_vs_right_branch(self, sf: np.ndarray) -> None:
        """Test that wider left branch yields lower sensitivity for sf below sf_peak."""
        sf_peak_val = 3.0
        cs_peak = np.array([[100.0]])
        sf_peak = np.array([[sf_peak_val]])
        width_r = np.array([[1.28]])

        result_wide_l = np.asarray(predict_contrast_sensitivity(sf, cs_peak, sf_peak, 10.0, width_r))
        result_narrow_l = np.asarray(predict_contrast_sensitivity(sf, cs_peak, sf_peak, 0.1, width_r))

        # Wider left branch drops sensitivity faster → lower values left of peak
        left_frames = sf < sf_peak_val
        assert np.all(result_wide_l[0, left_frames] < result_narrow_l[0, left_frames])


class TestPredictContrastResponse(TestSetup):
    """Tests for predict_contrast_response function."""

    @pytest.fixture
    def sensitivity(self) -> np.ndarray:
        """Contrast sensitivity for NUM_VOXELS voxels and NUM_FRAMES frames, shape (NUM_VOXELS, NUM_FRAMES)."""
        return np.full((self.num_voxels, self.num_frames), 100.0)

    @pytest.fixture
    def slope_crf(self) -> np.ndarray:
        """Naka-Rushton slope for NUM_VOXELS voxels, shape (NUM_VOXELS, 1)."""
        return np.array([[2.0], [2.0], [3.0]])

    def test_shape_error_sensitivity(self, contrast: np.ndarray, slope_crf: np.ndarray) -> None:
        """Test that ShapeError is raised when sensitivity has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_response(contrast, np.ones(self.num_frames), slope_crf)

    def test_shape_error_slope_crf(self, contrast: np.ndarray, sensitivity: np.ndarray) -> None:
        """Test that ShapeError is raised when slope_crf has fewer than 2 dimensions."""
        with pytest.raises(ShapeError):
            predict_contrast_response(contrast, sensitivity, np.array([2.0]))

    def test_batch_dimension_error(self, contrast: np.ndarray) -> None:
        """Test that BatchDimensionError is raised when sensitivity and slope_crf batch dims differ."""
        sensitivity_2 = np.ones((2, self.num_frames))
        slope_crf_3 = np.array([[2.0], [2.0], [2.0]])
        with pytest.raises(BatchDimensionError):
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
        result = np.asarray(predict_contrast_response(contrast, sensitivity, slope_crf))
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_known_naka_rushton_value(self) -> None:
        """Test Naka-Rushton formula against the analytical expression at a single operating point.

        At contrast c, sensitivity s, and slope q: R = c^q / (c^q + (1/s)^q).

        """
        c, s, q = 0.5, 100.0, 2.0
        contrast = np.array([c])
        sensitivity = np.array([[s]])
        slope_crf = np.array([[q]])

        result = float(np.asarray(predict_contrast_response(contrast, sensitivity, slope_crf)).flat[0])
        expected = c**q / (c**q + (100.0 / s) ** q)

        assert np.allclose(result, expected)


class TestCSFResponse(TestSetup):
    """Tests for CSFResponse class."""

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
        model = CSFResponse()
        assert model.parameter_names == ["cs_peak", "sf_peak", "width_r", "slope_crf"]

    @parametrize_dtype
    def test_predict(self, stimulus: CSFStimulus, params: pd.DataFrame, dtype: str) -> None:
        """Test that response prediction returns correct shape."""
        model = CSFResponse()
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)


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
        model = CSFModel(impulse_model=None, temporal_model=None)
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)

    @parametrize_dtype
    def test_simple_csf_model_predict(self, stimulus: CSFStimulus, params: pd.DataFrame, dtype: str) -> None:
        """Test that SimpleCSFModel with CSFResponse returns correct shape."""
        model = SimpleCSFModel(csf_model=CSFResponse())
        result = model(stimulus, params, dtype=dtype)
        assert result.shape == (self.num_voxels, self.num_frames)
