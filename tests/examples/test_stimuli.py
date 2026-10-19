"""Test bundled example stimuli."""

from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat
from prfmodel.examples import load_1d_prf_lognumerosity_stimulus
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.examples._stimuli import _load_chn_aperture_stimulus
from prfmodel.examples._stimuli import _resample_frames
from prfmodel.stimuli import PRFStimulus


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


def test_load_1d_prf_lognumerosity_stimulus():
    """Test that load_1d_prf_lognumerosity_stimulus returns a 1D stimulus."""
    stimulus = load_1d_prf_lognumerosity_stimulus()

    expected_ndim = 2

    assert isinstance(stimulus, PRFStimulus)
    assert len(stimulus.design.shape) == expected_ndim  # (num_frames, num_coordinates)
    assert len(stimulus.grid.shape) == expected_ndim  # (num_coordinates, 1)
    assert np.all(stimulus.design.sum(axis=1) == 1)  # Check one-hot encoding
    np.testing.assert_allclose(stimulus.grid[:, 0], np.log([1, 2, 3, 4, 5, 6, 7, 20]))  # Check unique log numerosities


def test_resample_frames_weights_by_overlap():
    """Test that frames of 1 s are averaged into frames of 1.2 s by the time they overlap each target frame."""
    design = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    resampled = _resample_frames(design, source_duration=1.0, target_duration=1.2, num_frames=5)

    # Target frame 0 spans 0-1.2 s: 1 s of frame 0 and 0.2 s of frame 1; target frame 4 spans 4.8-6 s
    np.testing.assert_allclose(resampled, [1 / 1.2, 0.0, 0.0, 0.0, 1 / 1.2])


def test_resample_frames_keeps_spatial_axes():
    """Test that resampling acts on the first axis only and keeps a constant design constant."""
    design = np.ones((366, 4, 3))

    resampled = _resample_frames(design, source_duration=1.0, target_duration=1.2, num_frames=305)

    assert resampled.shape == (305, 4, 3)
    np.testing.assert_allclose(resampled, 1.0)


def test_resample_frames_too_short():
    """Test that a design that does not cover the requested frames is refused."""
    with pytest.raises(ValueError, match="does not cover"):
        _resample_frames(np.ones(10), source_duration=1.0, target_duration=1.2, num_frames=10)


def test_load_chn_aperture_stimulus(tmp_path: Path):
    """Test that a CHN aperture file is turned into a stimulus with the frames first and a (y, x) grid."""
    num_rows, num_cols, num_frames = 3, 4, 12
    x, y = np.meshgrid(np.linspace(-8, 8, num_cols), np.linspace(8, -8, num_rows))
    img = np.zeros((num_rows, num_cols, num_frames), dtype=np.uint8)
    img[0, 1, :6] = 1
    path = tmp_path / "aperture.mat"
    savemat(path, {"stim": {"x": x, "y": y, "t": np.arange(num_frames), "img": img}})

    stimulus = _load_chn_aperture_stimulus(path, num_frames=10)

    assert stimulus.design.shape == (10, num_rows, num_cols)
    assert stimulus.dimension_labels == ["y", "x"]
    np.testing.assert_allclose(stimulus.grid[..., 0], y)
    np.testing.assert_allclose(stimulus.grid[..., 1], x)
    np.testing.assert_allclose(stimulus.design[:5, 0, 1], 1.0)
    assert stimulus.design[:, 1:].max() == 0.0
