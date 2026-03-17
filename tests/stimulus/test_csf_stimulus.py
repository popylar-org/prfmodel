"""Tests for CSFStimulus and PRFCSFStimulus."""

import numpy as np
import pytest
from prfmodel.stimuli.csf import CSFStimulus
from prfmodel.stimuli.csf import CSFStimulusDimensionError
from prfmodel.stimuli.csf import CSFStimulusShapeError


@pytest.fixture
def csf_stimulus():
    """CSF stimulus object."""
    return CSFStimulus(
        sf=np.array([1.0, 3.0, 6.0, 12.0]),
        contrast=np.array([0.1, 0.2, 0.4, 0.8]),
    )


def test_csf_stimulus_construction(csf_stimulus: CSFStimulus):
    """Test that sf and contrast arrays are stored with correct shape."""
    assert csf_stimulus.sf.shape == (4,)
    assert csf_stimulus.contrast.shape == (4,)


def test_csf_stimulus_sf_dimension_error():
    """Test that CSFStimulusDimensionError is raised when sf is not 1D."""
    with pytest.raises(CSFStimulusDimensionError):
        CSFStimulus(
            sf=np.zeros((2, 2)),
            contrast=np.zeros(4),
        )


def test_csf_stimulus_contrast_dimension_error():
    """Test that CSFStimulusDimensionError is raised when contrast is not 1D."""
    with pytest.raises(CSFStimulusDimensionError):
        CSFStimulus(
            sf=np.zeros(4),
            contrast=np.zeros((4, 1)),
        )


def test_csf_stimulus_shape_mismatch_error():
    """Test that CSFStimulusShapeError is raised when sf and contrast have different lengths."""
    with pytest.raises(CSFStimulusShapeError):
        CSFStimulus(
            sf=np.zeros(4),
            contrast=np.zeros(5),
        )


def test_csf_stimulus_str(csf_stimulus: CSFStimulus):
    """Test that the string representation includes the class name and field names."""
    s = str(csf_stimulus)
    assert "CSFStimulus" in s
    assert "sf" in s
    assert "contrast" in s


def test_csf_stimulus_eq():
    """Test that two CSFStimulus objects with identical arrays compare as equal."""
    sf = np.array([1.0, 3.0])
    contrast = np.array([0.1, 0.5])
    s1 = CSFStimulus(sf=sf, contrast=contrast)
    s2 = CSFStimulus(sf=sf.copy(), contrast=contrast.copy())
    assert s1 == s2


def test_csf_stimulus_neq_different_values():
    """Test that two CSFStimulus objects with different sf arrays compare as not equal."""
    s1 = CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.5]))
    s2 = CSFStimulus(sf=np.array([1.0, 6.0]), contrast=np.array([0.1, 0.5]))
    assert s1 != s2
