"""Tests for response, evaluation, parameter, and surface plotting functions."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from nilearn.surface import InMemoryMesh
from nilearn.surface import PolyMesh
from prfmodel.plotting import plot_1d_prf_stimulus
from prfmodel.plotting import plot_2d_prf_centers
from prfmodel.plotting import plot_grid_parameter_distribution
from prfmodel.plotting import plot_observed_predicted
from prfmodel.plotting import plot_parameter_by_roi
from prfmodel.plotting import plot_r_squared_comparison
from prfmodel.plotting import plot_r_squared_hist
from prfmodel.plotting import plot_response_heatmap
from prfmodel.plotting import plot_surface_stat_map
from prfmodel.plotting._utils import _grid_bin_edges
from prfmodel.stimuli import PRFStimulus


@pytest.fixture(autouse=True)
def close_figures():
    """Auto-close all figures."""
    yield
    plt.close("all")


@pytest.fixture
def rng():
    """Random number generator."""
    return np.random.default_rng(0)


def _assert_figure_axes(fig: mpl.figure.Figure, ax: mpl.axes.Axes) -> None:
    assert isinstance(fig, mpl.figure.Figure), "Does not create the Figure type"
    assert isinstance(ax, mpl.axes.Axes), "Does not create the Axes type"


def test__grid_bin_edges():
    """Test that bin edges are centered on regularly spaced grid values."""
    np.testing.assert_allclose(_grid_bin_edges([0.0, 1.0, 2.0]), [-0.5, 0.5, 1.5, 2.5])


def test__grid_bin_edges_log():
    """Test that bin edges are centered on log-spaced grid values on a log scale."""
    edges = _grid_bin_edges([1.0, 10.0, 100.0], log=True)
    np.testing.assert_allclose(edges, [10**-0.5, 10**0.5, 10**1.5, 10**2.5])


@pytest.mark.parametrize(("values", "log"), [([0.0, 1.0, 3.0], False), ([1.0], False), ([0.0, 1.0], True)])
def test__grid_bin_edges_invalid(values: list[float], log: bool):
    """Test that invalid grids raise an error."""
    with pytest.raises(ValueError):
        _grid_bin_edges(values, log=log)


def test_plot_response_heatmap(rng: np.random.Generator):
    """Test that the heatmap shows the response."""
    response = rng.normal(size=(20, 30))
    fig, ax = plot_response_heatmap(response)
    _assert_figure_axes(fig, ax)
    np.testing.assert_allclose(ax.images[0].get_array().data, response)
    assert len(fig.axes) == 2, "Colorbar is missing"  # noqa: PLR2004


def test_plot_response_heatmap_no_colorbar(rng: np.random.Generator):
    """Test that the colorbar can be turned off."""
    fig, _ = plot_response_heatmap(rng.normal(size=(5, 10)), colorbar_label=None)
    assert len(fig.axes) == 1, "Colorbar should not be drawn"


def test_plot_response_heatmap_wrong_ndim():
    """Test that a response that is not 2-dimensional raises an error."""
    with pytest.raises(ValueError, match="2-dimensional"):
        plot_response_heatmap(np.zeros(10))


def test_plot_observed_predicted():
    """Test that observed and multiple predicted timecourses are drawn."""
    observed = np.arange(10.0)
    predicted = {"Grid": observed * 0.5, "SGD": observed[np.newaxis] * 0.9}
    fig, ax = plot_observed_predicted(observed, predicted, split=5)
    _assert_figure_axes(fig, ax)
    # Observed, two predictions, and the split line
    assert len(ax.lines) == 4  # noqa: PLR2004
    np.testing.assert_allclose(ax.lines[2].get_ydata(), predicted["SGD"].squeeze())
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Observed", "Grid", "SGD"]


def test_plot_observed_predicted_existing_axes():
    """Test that an existing axes is used."""
    fig, axes = plt.subplots(1, 2)
    fig_out, ax_out = plot_observed_predicted(np.zeros(5), np.ones(5), ax=axes[1])
    assert fig_out is fig
    assert ax_out is axes[1]
    assert len(axes[1].lines) == 2  # noqa: PLR2004


def test_plot_observed_predicted_shape_mismatch():
    """Test that predictions with a different shape raise an error."""
    with pytest.raises(ValueError, match="must have shape"):
        plot_observed_predicted(np.zeros(5), np.zeros(6))


def test_plot_r_squared_hist(rng: np.random.Generator):
    """Test that the histogram counts the scores."""
    r_squared = rng.uniform(size=50)
    r_squared[0] = np.nan
    fig, ax = plot_r_squared_hist(r_squared, bins=10, value_range=(0.0, 1.0))
    _assert_figure_axes(fig, ax)
    assert sum(patch.get_height() for patch in ax.patches) == 49  # noqa: PLR2004
    assert ax.get_legend() is None


def test_plot_r_squared_hist_clip():
    """Test that clipping counts scores outside the range in the outermost bins."""
    r_squared = np.array([-2.0, 0.5, 2.0])
    _, ax = plot_r_squared_hist(r_squared, bins=2, value_range=(0.0, 1.0), clip=True)
    assert [patch.get_height() for patch in ax.patches] == [1, 2]


def test_plot_r_squared_hist_mapping(rng: np.random.Generator):
    """Test that multiple sets of scores are labeled in a legend."""
    _, ax = plot_r_squared_hist({"train": rng.uniform(size=10), "test": rng.uniform(size=10)}, log=True)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["train", "test"]
    assert ax.get_yscale() == "log"


def test_plot_r_squared_comparison(rng: np.random.Generator):
    """Test that the comparison counts all valid units."""
    r_squared_x = rng.uniform(size=50)
    r_squared_y = rng.uniform(size=50)
    r_squared_y[0] = np.nan
    fig, ax = plot_r_squared_comparison(r_squared_x, r_squared_y, value_range=(0.0, 1.0))
    _assert_figure_axes(fig, ax)
    assert ax.collections[0].get_array().sum() == 49  # noqa: PLR2004


def test_plot_r_squared_comparison_shape_mismatch():
    """Test that scores with different shapes raise an error."""
    with pytest.raises(ValueError, match="same shape"):
        plot_r_squared_comparison(np.zeros(5), np.zeros(6))


def test_plot_grid_parameter_distribution():
    """Test that estimates are counted per grid value and edges are highlighted."""
    parameter_values = {"sigma": np.array([1.0, 2.0, 4.0, 8.0])}
    parameters = pd.DataFrame({"sigma": [1.0, 2.0, 2.0, 4.0, 8.0, 8.0, 8.0]})
    fig, ax = plot_grid_parameter_distribution(parameters, parameter_values, "sigma", log=True)
    _assert_figure_axes(fig, ax)
    inner, edge = ax.patches
    np.testing.assert_allclose(inner.get_data().values, [0, 2, 1, 0])
    np.testing.assert_allclose(edge.get_data().values, [1, 0, 0, 3])
    assert ax.get_xscale() == "log"


def test_plot_parameter_by_roi():
    """Test that means and errors are computed per ROI in the given order."""
    parameters = pd.DataFrame({"sigma": [1.0, 3.0, 10.0, 10.0, 5.0]})
    roi = ["V1", "V1", "V2", "V2", "V3"]
    fig, ax = plot_parameter_by_roi(parameters, "sigma", roi, order=["V2", "V1"], error="sem")
    _assert_figure_axes(fig, ax)
    container = ax.containers[0]
    np.testing.assert_allclose(np.asarray(container.lines[0].get_ydata(), dtype=float), [10.0, 2.0])
    assert [label.get_text() for label in ax.get_xticklabels()] == ["V2", "V1"]


def test_plot_parameter_by_roi_invalid():
    """Test that invalid arguments raise errors."""
    parameters = pd.DataFrame({"sigma": [1.0, 2.0]})
    with pytest.raises(ValueError, match="one label"):
        plot_parameter_by_roi(parameters, "sigma", ["V1"])
    with pytest.raises(ValueError, match="'error'"):
        plot_parameter_by_roi(parameters, "sigma", ["V1", "V1"], error="var")  # type: ignore[arg-type]


def test_plot_2d_prf_centers():
    """Test that pRF centers are counted on the grid and the stimulus is marked."""
    parameter_values = {"mu_x": np.linspace(-2.0, 2.0, 5), "mu_y": np.linspace(-1.0, 1.0, 3)}
    parameters = pd.DataFrame({"mu_x": [-2.0, -2.0, 1.0, np.nan], "mu_y": [-1.0, -1.0, 0.0, 0.0]})
    stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=5, width=8, height=8)
    fig, ax = plot_2d_prf_centers(parameters, parameter_values, stimulus=stimulus)
    _assert_figure_axes(fig, ax)
    counts = ax.collections[0].get_array()
    assert counts.shape == (3, 5)
    assert counts[0, 0] == 2  # noqa: PLR2004
    assert counts[1, 3] == 1
    assert counts.sum() == 3  # noqa: PLR2004
    assert len(ax.patches) == 1, "Stimulus rectangle is missing"


def test_plot_2d_prf_centers_1d_stimulus():
    """Test that a stimulus that is not 2-dimensional raises an error."""
    stimulus = PRFStimulus(design=np.zeros((3, 4)), grid=np.zeros((4, 1)))
    with pytest.raises(ValueError, match="2-dimensional"):
        plot_2d_prf_centers(pd.DataFrame({"mu_x": [0.0], "mu_y": [0.0]}), stimulus=stimulus)


@pytest.fixture
def stimulus_1d():
    """One-dimensional stimulus."""
    design = np.eye(4)[[0, 1, 2, 3, 2, 1]]
    grid = np.log(np.array([1.0, 2.0, 3.0, 4.0]))[:, np.newaxis]
    return PRFStimulus(design=design, grid=grid, dimension_labels=["log(n)"])


def test_plot_1d_prf_stimulus(stimulus_1d: PRFStimulus):
    """Test that the design and tick labels are shown."""
    fig, ax = plot_1d_prf_stimulus(stimulus_1d, secondary_tick_labels=[1, 2, 3, 4], secondary_ylabel="n")
    _assert_figure_axes(fig, ax)
    np.testing.assert_allclose(ax.images[0].get_array().data, stimulus_1d.design.T)
    assert [label.get_text() for label in ax.get_yticklabels()] == ["0.00", "0.69", "1.10", "1.39"]
    assert ax.get_ylabel() == "log(n)"


def test_plot_1d_prf_stimulus_invalid(stimulus_1d: PRFStimulus):
    """Test that invalid stimuli and tick labels raise errors."""
    with pytest.raises(ValueError, match="1-dimensional"):
        plot_1d_prf_stimulus(PRFStimulus.create_2d_bar_stimulus(num_frames=5, width=8, height=8))
    with pytest.raises(ValueError, match="tick labels"):
        plot_1d_prf_stimulus(stimulus_1d, tick_labels=["a"])


def test_plot_surface_stat_map():
    """Test that a stat map is plotted on a mesh of both hemispheres."""
    coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    mesh = PolyMesh(left=InMemoryMesh(coordinates, faces), right=InMemoryMesh(coordinates + 2.0, faces))
    fig, ax = plot_surface_stat_map(mesh, np.array([0.0, 1.0, 2.0, -1.0, -2.0, -3.0]), cyclic=True, title="Angle")
    assert isinstance(fig, mpl.figure.Figure)
    assert ax.name == "3d"
