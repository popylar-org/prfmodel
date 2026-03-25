"""Tests for CSFStimulus and PRFCSFStimulus."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from prfmodel.stimuli.csf import CSFStimulus
from prfmodel.stimuli.csf import CSFStimulusDimensionError
from prfmodel.stimuli.csf import CSFStimulusShapeError
from prfmodel.stimuli.csf import plot_csf_stimulus_curve
from prfmodel.stimuli.csf import plot_csf_stimulus_design

mpl.use("Agg")


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


class TestPlotCSFStimulusDesign:
    """Tests for plot_csf_stimulus_design."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        """Auto-close all figure."""
        yield
        plt.close("all")

    def test_returns_figure_and_axes(self, csf_stimulus: CSFStimulus):
        """Test that the function returns a Figure and Axes."""
        fig, ax = plot_csf_stimulus_design(csf_stimulus)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)

    def test_one_line_per_sf_level(self, csf_stimulus: CSFStimulus):
        """Test that the number of plotted lines matches the number of unique spatial frequencies."""
        _, ax = plot_csf_stimulus_design(csf_stimulus)
        n_unique_sf = len(np.unique(csf_stimulus.sf))
        assert len(ax.lines) == n_unique_sf

    def test_axis_labels(self, csf_stimulus: CSFStimulus):
        """Test that axis labels are set."""
        _, ax = plot_csf_stimulus_design(csf_stimulus)
        assert ax.get_xlabel() == "Time frame"
        assert ax.get_ylabel() == "Contrast"

    def test_kwargs_forwarded_to_subplots(self, csf_stimulus: CSFStimulus):
        """Test that kwargs are forwarded to plt.subplots (e.g. figsize)."""
        fig, _ = plot_csf_stimulus_design(csf_stimulus, figsize=(6, 3))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(3.0)


class TestPlotCSFStimulusCurve:
    """Tests for plot_csf_stimulus_curve."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        """Auto-close all figure."""
        yield
        plt.close("all")

    @pytest.fixture
    def csf_curve(self, csf_stimulus: CSFStimulus):
        """Contrast sensitivity curve."""
        return np.full(csf_stimulus.contrast.shape[0], 50.0)

    def test_returns_figure_and_axes(self, csf_stimulus: CSFStimulus, csf_curve: np.ndarray):
        """Test that the function returns a Figure and Axes."""
        fig, ax = plot_csf_stimulus_curve(csf_stimulus, csf_curve)
        assert isinstance(fig, mpl.figure.Figure)
        assert isinstance(ax, mpl.axes.Axes)

    def test_axis_labels(self, csf_stimulus: CSFStimulus, csf_curve: np.ndarray):
        """Test that axis labels are set."""
        _, ax = plot_csf_stimulus_curve(csf_stimulus, csf_curve)
        assert ax.get_xlabel() == "Spatial frequency"
        assert ax.get_ylabel() == "Contrast sensitivity"

    def test_multi_voxel_csf(self, csf_stimulus: CSFStimulus):
        """Test that a 2-D csf array (multiple voxels) is accepted."""
        num_voxels = 3
        csf_2d = np.full((num_voxels, csf_stimulus.contrast.shape[0]), 50.0)
        fig, _ = plot_csf_stimulus_curve(csf_stimulus, csf_2d)
        assert isinstance(fig, mpl.figure.Figure)

    def test_frame_mismatch_raises_value_error(self, csf_stimulus: CSFStimulus):
        """Test that a ValueError is raised when csf has a different number of frames than stimulus."""
        wrong_csf = np.full(csf_stimulus.contrast.shape[0] + 1, 50.0)
        with pytest.raises(ValueError, match="same number of frames"):
            plot_csf_stimulus_curve(csf_stimulus, wrong_csf)

    def test_kwargs_forwarded_to_subplots(self, csf_stimulus: CSFStimulus, csf_curve: np.ndarray):
        """Test that kwargs are forwarded to plt.subplots (e.g. figsize)."""
        fig, _ = plot_csf_stimulus_curve(csf_stimulus, csf_curve, figsize=(8, 4))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8.0)
        assert h == pytest.approx(4.0)
