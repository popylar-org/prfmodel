"""Tests for stimulus plotting functions."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from prfmodel.plotting._stimuli import _get_grid_limits
from prfmodel.plotting._stimuli import animate_2d_prf_stimulus
from prfmodel.plotting._stimuli import plot_2d_prf_stimulus
from prfmodel.plotting._stimuli import plot_csf_stimulus_curve
from prfmodel.plotting._stimuli import plot_csf_stimulus_design
from prfmodel.stimuli import CSFStimulus
from prfmodel.stimuli import PRFStimulus


def test__get_grid_limits():
    """Test that grid limits get extracted correctly."""
    x = np.arange(-4, 4, 1)
    y = np.arange(-2, 6, 2)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack((xv, yv), axis=-1)
    result = _get_grid_limits(grid)
    expected = (-4.0, 3.0, -2.0, 4.0)
    assert result == expected, "Grid extent extracted incorrectly"


@pytest.fixture
def bar_stimulus():
    """Create bar stimulus to plot."""
    return PRFStimulus.create_2d_bar_stimulus(num_frames=100, width=128, height=64)


def test_animate_2d_stimulus(bar_stimulus: PRFStimulus):
    """Test that animation uses the correct input data."""
    ani = animate_2d_prf_stimulus(bar_stimulus)
    assert isinstance(ani, animation.ArtistAnimation), "Wrong type returned"
    reconstructed = np.stack([frame[0].get_array().data for frame in ani._framedata])  # noqa: SLF001
    np.testing.assert_allclose(reconstructed, bar_stimulus.design, err_msg="Animation uses wrong data")


def test_plot_2d_stimulus(bar_stimulus: PRFStimulus):
    """Test that plotting uses the correct input data."""
    frame_idx = 10
    fig, ax = plot_2d_prf_stimulus(bar_stimulus, frame_idx)
    assert isinstance(fig, mpl.figure.Figure), "Does not create the Figure type"
    assert isinstance(ax, mpl.axes.Axes), "Does not create the Axes type"
    img = ax.images[0]

    plotted_data = img.get_array().data
    np.testing.assert_allclose(plotted_data, bar_stimulus.design[frame_idx], err_msg="Figure uses wrong data")


class CSFPlottingSetup:
    """Setup for CSF stimulus plotting tests."""

    @pytest.fixture(autouse=True)
    def close_figures(self):
        """Auto-close all figure."""
        yield
        plt.close("all")

    @pytest.fixture
    def csf_stimulus(self):
        """CSF stimulus object."""
        return CSFStimulus(
            sf=np.array([1.0, 3.0, 6.0, 12.0]),
            contrast=np.array([0.1, 0.2, 0.4, 0.8]),
        )


class TestPlotCSFStimulusDesign(CSFPlottingSetup):
    """Tests for plot_csf_stimulus_design."""

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


class TestPlotCSFStimulusCurve(CSFPlottingSetup):
    """Tests for plot_csf_stimulus_curve."""

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
