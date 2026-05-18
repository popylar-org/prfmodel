"""Tests for stimulus plotting functions."""

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import animation
from prfmodel.plotting._stimuli import _get_grid_limits
from prfmodel.plotting._stimuli import animate_2d_prf_stimulus
from prfmodel.plotting._stimuli import plot_2d_prf_stimulus
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
