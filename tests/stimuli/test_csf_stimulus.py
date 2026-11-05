"""Tests for CSFStimulus and PRFCSFStimulus."""

import numpy as np
import pytest
from prfmodel.stimuli._csf import CSFStimulus
from prfmodel.stimuli._csf import CSFStimulusDimensionError
from prfmodel.stimuli._csf import CSFStimulusShapeError


@pytest.fixture
def csf_stimulus():
    """CSF stimulus object."""
    return CSFStimulus(
        sf=np.array([1.0, 3.0, 6.0, 12.0]),
        contrast=np.array([0.1, 0.2, 0.4, 0.8]),
        frame_duration=1.0,
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
            frame_duration=1.0,
        )


def test_csf_stimulus_contrast_dimension_error():
    """Test that CSFStimulusDimensionError is raised when contrast is not 1D."""
    with pytest.raises(CSFStimulusDimensionError):
        CSFStimulus(
            sf=np.zeros(4),
            contrast=np.zeros((4, 1)),
            frame_duration=1.0,
        )


def test_csf_stimulus_shape_mismatch_error():
    """Test that CSFStimulusShapeError is raised when sf and contrast have different lengths."""
    with pytest.raises(CSFStimulusShapeError):
        CSFStimulus(
            sf=np.zeros(4),
            contrast=np.zeros(5),
            frame_duration=1.0,
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
    s1 = CSFStimulus(sf=sf, contrast=contrast, frame_duration=1.0)
    s2 = CSFStimulus(sf=sf.copy(), contrast=contrast.copy(), frame_duration=1.0)
    assert s1 == s2


def test_csf_stimulus_neq_different_values():
    """Test that two CSFStimulus objects with different sf arrays compare as not equal."""
    s1 = CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.5]), frame_duration=1.0)
    s2 = CSFStimulus(sf=np.array([1.0, 6.0]), contrast=np.array([0.1, 0.5]), frame_duration=1.0)
    assert s1 != s2


def test_csf_stimulus_frame_duration_is_required():
    """Test that a stimulus cannot be created without a frame duration."""
    with pytest.raises(TypeError, match="frame_duration"):
        CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.5]))  # type: ignore[call-arg]


@pytest.mark.parametrize("frame_duration", [0.0, -1.5])
def test_csf_stimulus_frame_duration_must_be_positive(frame_duration: float):
    """Test that a non-positive frame duration is rejected."""
    with pytest.raises(ValueError, match="'frame_duration' must be > 0"):
        CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.5]), frame_duration=frame_duration)


def test_csf_stimulus_neq_different_frame_duration():
    """Test that two CSFStimulus objects with different frame durations compare as not equal."""
    s1 = CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.5]), frame_duration=1.0)
    s2 = CSFStimulus(sf=np.array([1.0, 3.0]), contrast=np.array([0.1, 0.5]), frame_duration=1.5)
    assert s1 != s2


def test_csf_stimulus_to_tensors_keeps_frame_duration_as_float(csf_stimulus: CSFStimulus):
    """Test that the frame duration stays a Python float so that it is a constant when traced."""
    tensors = csf_stimulus.to_tensors()

    assert type(tensors.frame_duration) is float
    assert tensors.frame_duration == csf_stimulus.frame_duration
