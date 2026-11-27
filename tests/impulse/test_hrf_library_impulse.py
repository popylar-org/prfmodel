"""Tests for the canonical HRF library impulse response."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters import GridFitter
from prfmodel.impulse import HRFLibraryImpulse
from prfmodel.impulse._hrf_library import HRF_LIBRARY_RESOLUTION
from prfmodel.impulse._hrf_library import HRF_LIBRARY_SIZE
from prfmodel.impulse._hrf_library import _load_hrf_library
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup
from tests.models.conftest import parametrize_dtype
from .conftest import TestImpulseSetup


class TestHRFLibraryImpulse(TestImpulseSetup):
    """Tests for HRFLibraryImpulse class."""

    @pytest.fixture
    def parameters(self):
        """Every HRF in the library."""
        return pd.DataFrame({"hrf_index": np.arange(HRF_LIBRARY_SIZE, dtype=float)})

    @pytest.fixture
    def irf_model(self):
        """Impulse model object."""
        return HRFLibraryImpulse(self.duration, self.offset, self.resolution, self.norm)

    @pytest.fixture
    def irf_model_default(self):
        """Impulse model object with default parameters."""
        return HRFLibraryImpulse(self.duration, self.offset, self.resolution, self.norm, {"hrf_index": 9.0})

    def test_default_duration_covers_library(self):
        """Test that the default duration covers the full 50 seconds of the library."""
        library = _load_hrf_library()

        assert HRFLibraryImpulse().duration == pytest.approx((library.shape[1] - 1) * HRF_LIBRARY_RESOLUTION)

    def test_library_matches_file_at_native_resolution(self):
        """Test that sampling at the native resolution reproduces the library without interpolation error."""
        library = _load_hrf_library()
        model = HRFLibraryImpulse(duration=50.0, resolution=HRF_LIBRARY_RESOLUTION, norm=None)

        resampled = np.asarray(model.get_library("float64"))

        assert resampled.shape == (HRF_LIBRARY_SIZE, model.num_frames)
        np.testing.assert_allclose(resampled, library[:, : model.num_frames], atol=1e-12)

    def test_library_is_resampled_at_resolution(self):
        """Test that a coarser resolution picks every n-th sample of the library."""
        library = _load_hrf_library()
        model = HRFLibraryImpulse(duration=50.0, resolution=1.0, norm=None)

        resampled = np.asarray(model.get_library("float64"))

        np.testing.assert_allclose(resampled, library[:, ::10][:, : model.num_frames], atol=1e-12)

    def test_response_beyond_library_is_zero(self):
        """Test that frames after the end of the library are zero rather than extrapolated."""
        model = HRFLibraryImpulse(duration=60.0, resolution=1.0, norm=None)

        resampled = np.asarray(model.get_library("float64"))
        times = np.asarray(model.get_frames("float64"))[0]

        assert np.all(resampled[:, times > 50.0] == 0.0)  # noqa: PLR2004 (end of the library)

    def test_library_cache_is_not_shared_across_resolutions(self):
        """Test that a copy from 'at_resolution' does not read the library cached at another resolution."""
        model = HRFLibraryImpulse()
        params = pd.DataFrame({"hrf_index": [5.0]})

        # Fill the cache of the original model before the copy is made
        model(params)
        resolved = model.at_resolution(1.5)

        assert resolved(params).shape == (1, resolved.num_frames)
        np.testing.assert_array_equal(resolved(params), HRFLibraryImpulse(resolution=1.5)(params))

    @parametrize_dtype
    def test_call_selects_hrf(self, irf_model: HRFLibraryImpulse, dtype: str):
        """Test that each unit gets the normalized library row indicated by its 'hrf_index'."""
        hrf_index = np.array([19, 0, 7, 7])
        params = pd.DataFrame({"hrf_index": hrf_index.astype(float)})

        resp = irf_model(params, dtype)
        library = np.asarray(irf_model.get_library("float64"))[hrf_index]
        expected = library / library.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(resp, expected, rtol=1e-5, atol=1e-7)

    def test_hrfs_differ(self, irf_model: HRFLibraryImpulse, parameters: pd.DataFrame):
        """Test that no two HRFs in the library produce the same response."""
        resp = irf_model(parameters)

        assert len(np.unique(resp.round(8), axis=0)) == HRF_LIBRARY_SIZE

    @pytest.mark.parametrize(
        ("hrf_index", "match"),
        [
            (-1.0, "between 0 and 19"),
            (20.0, "between 0 and 19"),
            (2.5, "whole number"),
        ],
    )
    def test_invalid_hrf_index_raises(self, irf_model: HRFLibraryImpulse, hrf_index: float, match: str):
        """Test that an 'hrf_index' that does not index the library is rejected."""
        params = pd.DataFrame({"hrf_index": [0.0, hrf_index]})

        with pytest.raises(ValueError, match=match):
            irf_model(params)

    def test_invalid_default_hrf_index_raises(self):
        """Test that an invalid default 'hrf_index' is rejected when it is merged in."""
        model = HRFLibraryImpulse(default_parameters={"hrf_index": 25.0})

        with pytest.raises(ValueError, match="between 0 and 19"):
            model(pd.DataFrame(index=range(2)))


class TestHRFLibraryGridSearch(PRFStimulusSetup):
    """Tests for finding the best fitting HRF with a grid search."""

    @pytest.fixture
    def model(self):
        """Gaussian 2D pRF model with the HRF library impulse model."""
        return Gaussian2DPRFModel(impulse_model=HRFLibraryImpulse(), scaling_model=None)

    @pytest.mark.parametrize("compile_step", [False, True])
    def test_grid_search_recovers_hrf(self, stimulus: PRFStimulus, model: Gaussian2DPRFModel, compile_step: bool):
        """Test that a grid search over 'hrf_index' recovers the HRF used to simulate each unit."""
        true_params = pd.DataFrame(
            {
                "mu_y": 0.5,
                "mu_x": -0.5,
                "sigma": 1.0,
                "hrf_index": np.arange(HRF_LIBRARY_SIZE, dtype=float),
            },
        )
        observed = model(stimulus, true_params, dtype="float64")

        fitter = GridFitter(model=model, stimulus=stimulus, compile_step=compile_step, dtype="float64")
        _, params = fitter.fit(
            observed,
            {
                "mu_y": [-0.5, 0.5],
                "mu_x": [-0.5, 0.5],
                "sigma": [1.0, 2.0],
                "hrf_index": np.arange(HRF_LIBRARY_SIZE),
            },
            batch_size=16,
        )

        np.testing.assert_array_equal(params["hrf_index"].to_numpy(), true_params["hrf_index"].to_numpy())
        np.testing.assert_array_equal(params[["mu_y", "mu_x", "sigma"]], true_params[["mu_y", "mu_x", "sigma"]])

    def test_grid_search_rejects_invalid_hrf_index(self, stimulus: PRFStimulus, model: Gaussian2DPRFModel):
        """Test that a grid containing an index outside the library raises before the search starts."""
        fitter = GridFitter(model=model, stimulus=stimulus)
        observed = np.zeros((1, stimulus.design.shape[0]))

        with pytest.raises(ValueError, match="between 0 and 19"):
            fitter.fit(observed, {"mu_y": [0.0], "mu_x": [0.0], "sigma": [1.0], "hrf_index": np.arange(21)})
