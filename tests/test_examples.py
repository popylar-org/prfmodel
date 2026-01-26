"""Tests for examples."""

import tempfile
import numpy as np
import pytest
from prfmodel.examples import download_surface
from prfmodel.examples import load_2d_bar_stimulus
from prfmodel.examples import load_brain_atlas
from prfmodel.examples import load_single_subject_fmri_data
from prfmodel.stimulus import Stimulus

_NUM_VERTICES = 118584
_NUM_FRAMES = 120
_MIN_ROI_IDX = 0
_MAX_ROI_IDX = 179


pytest_skip_examples = pytest.mark.examples


def test_load_2d_bar_stimulus():
    """Test that load_2d_bar_stimulus returns stimulus object."""
    stimulus = load_2d_bar_stimulus()
    assert isinstance(stimulus, Stimulus)


@pytest_skip_examples
def test_download_surface():
    """Test that download_surface returns the correct data shape."""
    cortex = pytest.importorskip("cortex")
    subject = "hcp_999999"

    download_surface(subject)

    vertex = cortex.db.get_surfinfo(subject)

    assert vertex.data.shape == (_NUM_VERTICES,)


@pytest_skip_examples
def test_download_surface_value_error():
    """Test that download_surface raises error for unknown subject identifier."""
    _ = pytest.importorskip("cortex")

    with pytest.raises(ValueError):
        download_surface("test")


@pytest_skip_examples
def test_load_single_subject_fmri_data():
    """Test that load_single_subject_fmri_data returns objects with the correct data shapes."""
    response, stimulus = load_single_subject_fmri_data(tempfile.tempdir)

    assert response.shape == (_NUM_VERTICES, _NUM_FRAMES)
    assert stimulus.design.shape == (_NUM_FRAMES, 100, 100)
    assert stimulus.grid.shape == (100, 100, 2)


@pytest_skip_examples
def test_load_single_subject_fmri_data_units():
    """Test that load_single_subject_fmri_data returns response data with the correct unit."""

    def subset_finite(x: np.ndarray) -> np.ndarray:
        return x[np.any(np.isfinite(x), axis=1)]

    response, _ = load_single_subject_fmri_data(tempfile.tempdir, unit="psc")
    response = subset_finite(response)
    assert np.allclose(response.mean(axis=1), 0.0, atol=1e-6, equal_nan=True)

    response, _ = load_single_subject_fmri_data(tempfile.tempdir, unit="z_score")
    response = subset_finite(response)
    assert np.allclose(response.mean(axis=1), 0.0, atol=1e-6, equal_nan=True)
    assert np.allclose(response.std(axis=1), 1.0, atol=1e-6, equal_nan=True)

    response, _ = load_single_subject_fmri_data(tempfile.tempdir, unit="raw")
    response = subset_finite(response)
    assert not np.allclose(response.mean(axis=1), 0.0, atol=1e-6, equal_nan=True)
    assert not np.allclose(response.std(axis=1), 1.0, atol=1e-6, equal_nan=True)


@pytest_skip_examples
@pytest.mark.parametrize("hemisphere", ["left", "right"])
def test_load_single_subject_fmri_data_hemispheres(hemisphere: str):
    """Test that load_single_subject_fmri_data returns response data with the hemisphere shape."""
    response, _ = load_single_subject_fmri_data(tempfile.tempdir, hemisphere=hemisphere)
    assert response.shape == (_NUM_VERTICES // 2, _NUM_FRAMES)


@pytest_skip_examples
def test_load_single_subject_fmri_data_value_error():
    """Test that load_single_subject_fmri_data raises error for unknown hemispheres and units."""
    with pytest.raises(ValueError):
        load_single_subject_fmri_data(tempfile.tempdir, hemisphere="test")

    with pytest.raises(ValueError):
        load_single_subject_fmri_data(tempfile.tempdir, unit="test")


@pytest_skip_examples
def test_load_brain_atlas():
    """Test that load_brain_atlas returns object with correct range and shape."""
    atlas = load_brain_atlas(tempfile.tempdir)

    assert atlas.shape == (_NUM_VERTICES,)
    assert atlas.min() == _MIN_ROI_IDX
    assert atlas.max() == _MAX_ROI_IDX


@pytest_skip_examples
@pytest.mark.parametrize("hemisphere", ["left", "right"])
def test_load_brain_atlas_hemispheres(hemisphere: str):
    """Test that load_brain_atlas returns object with correct hemisphere range and shape."""
    atlas = load_brain_atlas(tempfile.tempdir, hemisphere=hemisphere)
    atlas = np.mod(atlas, 180)  # hemisphere indices are symmetric

    assert atlas.shape == (_NUM_VERTICES // 2,)
    assert atlas.min() == _MIN_ROI_IDX
    assert atlas.max() == _MAX_ROI_IDX


@pytest_skip_examples
def test_load_brain_atlas_value_error():
    """Test that load_brain_atlas returns object with correct hemisphere range and shape."""
    with pytest.raises(ValueError):
        load_brain_atlas(tempfile.tempdir, hemisphere="test")
