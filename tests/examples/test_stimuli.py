"""Test bundled example stimuli."""

import numpy as np
from prfmodel.examples import load_1d_prf_lognumerosity_stimulus
from prfmodel.examples import load_2d_prf_bar_stimulus
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
