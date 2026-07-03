"""Tests for examples."""

import tempfile
import numpy as np
import pytest
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.examples import load_single_subject_fmri_data
from prfmodel.examples import load_surface_mesh
from prfmodel.stimuli import PRFStimulus

_NUM_VERTICES = 118584
_NUM_FRAMES = 120


pytest_skip_examples = pytest.mark.examples


def test_load_2d_bar_stimulus_single():
    """Test that load_2d_prf_bar_stimulus returns a single stimulus object by default."""
    stimulus = load_2d_prf_bar_stimulus()
    assert isinstance(stimulus, PRFStimulus)


def test_load_2d_bar_stimulus_train_test():
    """Test that load_2d_prf_bar_stimulus returns a train and test stimulus object when return_test is true."""
    stimulus_train, stimulus_test = load_2d_prf_bar_stimulus(return_test=True)
    assert isinstance(stimulus_train, PRFStimulus)
    assert isinstance(stimulus_test, PRFStimulus)
    assert stimulus_train.design.shape == stimulus_test.design.shape
    np.testing.assert_array_equal(stimulus_train.grid, stimulus_test.grid)


@pytest_skip_examples
@pytest.mark.parametrize("surface_type", ["flat", "inflated", "pia", "wm"])
def test_download_surface(surface_type: str):
    """Test that download_surface returns the correct data shape."""
    mesh = load_surface_mesh(
        dest_dir="data",
        surface_type=surface_type,
    )

    assert mesh.n_vertices == _NUM_VERTICES


@pytest_skip_examples
def test_download_surface_surface_type_error():
    """Test that download_surface raises error for unknown subject identifier."""
    with pytest.raises(ValueError):
        load_surface_mesh(
            dest_dir="data",
            surface_type="test",
        )


@pytest_skip_examples
def test_load_single_subject_fmri_data():
    """Test that load_single_subject_fmri_data returns objects with the correct data shapes."""
    response = load_single_subject_fmri_data(tempfile.tempdir)

    assert response.shape == (_NUM_VERTICES, _NUM_FRAMES)


@pytest_skip_examples
@pytest.mark.parametrize("hemisphere", ["left", "right"])
def test_load_single_subject_fmri_data_hemispheres(hemisphere: str):
    """Test that load_single_subject_fmri_data returns response data with the hemisphere shape."""
    response = load_single_subject_fmri_data(tempfile.tempdir, hemisphere=hemisphere)
    assert response.shape == (_NUM_VERTICES // 2, _NUM_FRAMES)


@pytest_skip_examples
def test_load_single_subject_fmri_data_hemisphere_error():
    """Test that load_single_subject_fmri_data raises error for unknown hemispheres."""
    with pytest.raises(ValueError):
        load_single_subject_fmri_data(tempfile.tempdir, hemisphere="test")
