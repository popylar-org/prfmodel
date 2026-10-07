"""Model parameter plotting functions."""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Literal
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import _EXPECTED_NDIM
from ._utils import _get_figure_axes
from ._utils import _grid_bin_edges


def plot_grid_parameter_distribution(  # noqa: PLR0913
    parameters: pd.DataFrame,
    parameter_values: Mapping[str, np.ndarray | Sequence[float]],
    name: str,
    log: bool = False,
    xlabel: str | None = None,
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot how the estimates of a parameter are distributed across the values of a parameter grid.

    Draws a histogram with one bin centered on each value in the grid. Estimates at the smallest or largest value of
    the grid are highlighted: For these units, the best fitting parameter might lie outside the grid. Useful to
    diagnose the results of a :class:`~prfmodel.fitters.GridFitter`.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Estimated parameters with one row per unit (e.g., returned by :meth:`prfmodel.fitters.GridFitter.fit`).
    parameter_values : Mapping[str, numpy.ndarray or Sequence[float]]
        Values of the parameter grid (e.g., passed to :meth:`prfmodel.fitters.GridFitter.fit`). The values for `name`
        must be regularly spaced (or log-spaced if `log=True`).
    name : str
        Name of the parameter to plot.
    log : bool, optional
        Whether the grid values are log-spaced. If `True`, the x-axis has a logarithmic scale.
    xlabel : str or None, optional
        Label of the x-axis. If `None`, `name` is used.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If `None`, a new figure and axes are created.
    **kwargs
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` if `ax` is `None`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Raises
    ------
    ValueError
        If the grid values are not regularly spaced (or log-spaced if `log=True`) or contain fewer than two values.

    Notes
    -----
    Estimates outside the range of the grid (e.g., after refining the grid estimates with
    :class:`~prfmodel.fitters.SGDFitter`) are not shown.

    See Also
    --------
    plot_2d_prf_centers : Distribution of pRF centers in a two-dimensional visual field.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> parameter_values = {"sigma": np.geomspace(0.1, 10.0, 10)}
    >>> parameters = pd.DataFrame({"sigma": np.random.default_rng(0).choice(parameter_values["sigma"], 100)})
    >>> fig, ax = plot_grid_parameter_distribution(parameters, parameter_values, "sigma", log=True)

    """
    edges = _grid_bin_edges(parameter_values[name], log=log)

    fig, ax = _get_figure_axes(ax, **kwargs)

    counts, _ = np.histogram(parameters[name], bins=edges)

    is_edge_bin = np.zeros(counts.shape, dtype=bool)
    is_edge_bin[[0, -1]] = True

    ax.stairs(np.where(is_edge_bin, 0, counts), edges, fill=True, color="tab:gray")
    ax.stairs(np.where(is_edge_bin, counts, 0), edges, fill=True, color="tab:red", label="Grid edge")

    if log:
        ax.set_xscale("log")

    ax.set_xlabel(name if xlabel is None else xlabel)
    ax.set_ylabel("Number of units")
    ax.legend()

    return fig, ax


def plot_parameter_by_roi(  # noqa: PLR0913
    parameters: pd.DataFrame,
    name: str,
    roi: np.ndarray | Sequence[str] | pd.Series,
    order: Sequence[str] | None = None,
    error: Literal["std", "sem"] = "std",
    xlabel: str = "ROI",
    ylabel: str | None = None,
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the average of a parameter for each region of interest (ROI).

    Draws the mean of the parameter across the units in each ROI with error bars.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Estimated parameters with one row per unit.
    name : str
        Name of the parameter to plot.
    roi : numpy.ndarray or Sequence[str] or pandas.Series
        ROI label of each unit with shape `(num_units,)`.
    order : Sequence[str] or None, optional
        Order of the ROIs on the x-axis. ROIs that are not in `order` are not shown. If `None`, ROIs are shown in the
        order in which they first appear in `roi`.
    error : {"std", "sem"}, optional
        Whether the error bars show the standard deviation (`"std"`) or the standard error of the mean (`"sem"`) across
        the units in each ROI.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str or None, optional
        Label of the y-axis. If `None`, `name` is used.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If `None`, a new figure and axes are created.
    **kwargs
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` if `ax` is `None`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Raises
    ------
    ValueError
        If `roi` does not have one label for each unit or `error` is not `"std"` or `"sem"`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> parameters = pd.DataFrame({"sigma": np.random.default_rng(0).uniform(size=6)})
    >>> roi = ["V1", "V1", "V2", "V2", "V3", "V3"]
    >>> fig, ax = plot_parameter_by_roi(parameters, "sigma", roi, error="sem")

    """
    if error not in ("std", "sem"):
        msg = f"'error' must be 'std' or 'sem', but is '{error}'"
        raise ValueError(msg)

    roi = np.asarray(roi)

    if roi.shape != (parameters.shape[0],):
        msg = f"'roi' must have one label for each of the {parameters.shape[0]} units, but has shape {roi.shape}"
        raise ValueError(msg)

    if order is None:
        order = list(pd.unique(roi))

    grouped = pd.Series(np.asarray(parameters[name]), index=roi).groupby(level=0)
    stats = grouped.agg(["mean", error]).reindex(list(order))

    fig, ax = _get_figure_axes(ax, **kwargs)

    ax.errorbar([str(label) for label in order], stats["mean"], yerr=stats[error], fmt="o", capsize=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(name if ylabel is None else ylabel)

    return fig, ax


def plot_2d_prf_centers(  # noqa: PLR0913
    parameters: pd.DataFrame,
    parameter_values: Mapping[str, np.ndarray | Sequence[float]] | None = None,
    stimulus: PRFStimulus | None = None,
    x: str = "mu_x",
    y: str = "mu_y",
    bins: int = 50,
    cmap: str = "inferno",
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot how the centers of two-dimensional population receptive fields (pRFs) are distributed in the visual field.

    Draws a two-dimensional histogram of the pRF centers with a logarithmic color scale. Optionally, marks the part
    of the visual field that is covered by the stimulus.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Estimated parameters with one row per unit.
    parameter_values : Mapping[str, numpy.ndarray or Sequence[float]] or None, optional
        Values of the parameter grid (e.g., passed to :meth:`prfmodel.fitters.GridFitter.fit`). If given, the
        histogram has one bin centered on each grid value of `x` and `y`, which must be regularly spaced. If `None`,
        `bins` bins spanning the range of the estimates are used in each dimension.
    stimulus : PRFStimulus or None, optional
        Two-dimensional stimulus whose grid extent is marked by a rectangle.
    x : str, optional
        Name of the parameter for the horizontal pRF center coordinate.
    y : str, optional
        Name of the parameter for the vertical pRF center coordinate.
    bins : int, optional
        Number of bins in each dimension if `parameter_values` is `None`.
    cmap : str, optional
        Name of the colormap.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If `None`, a new figure and axes are created.
    **kwargs
        Keyword arguments passed to :func:`matplotlib.pyplot.subplots` if `ax` is `None`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    Raises
    ------
    ValueError
        If the stimulus is not 2-dimensional.

    See Also
    --------
    plot_grid_parameter_distribution : Distribution of a single parameter across grid values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> parameters = pd.DataFrame({"mu_x": rng.normal(size=100), "mu_y": rng.normal(size=100)})
    >>> stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=10, width=32, height=32)
    >>> fig, ax = plot_2d_prf_centers(parameters, stimulus=stimulus)

    """
    if stimulus is not None and stimulus.grid.shape[-1] != _EXPECTED_NDIM:
        msg = f"Stimulus must be 2-dimensional, but has {stimulus.grid.shape[-1]} dimensions"
        raise ValueError(msg)

    centers_x = np.asarray(parameters[x], dtype=float)
    centers_y = np.asarray(parameters[y], dtype=float)
    is_valid = ~(np.isnan(centers_x) | np.isnan(centers_y))

    if parameter_values is not None:
        hist_bins: int | list[np.ndarray] = [
            _grid_bin_edges(parameter_values[x]),
            _grid_bin_edges(parameter_values[y]),
        ]
    else:
        hist_bins = bins

    counts, x_edges, y_edges = np.histogram2d(centers_x[is_valid], centers_y[is_valid], bins=hist_bins)

    fig, ax = _get_figure_axes(ax, **kwargs)

    # Mask empty bins because they cannot be shown on a logarithmic color scale
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_equal(counts.T, 0), norm=LogNorm(), cmap=cmap)
    fig.colorbar(mesh, ax=ax, label="Number of units")

    if stimulus is not None:
        # The grid stores y in grid[..., 0] and x in grid[..., 1]
        y_min, x_min = stimulus.grid.min(axis=(0, 1))
        y_max, x_max = stimulus.grid.max(axis=(0, 1))
        ax.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                fill=False,
                edgecolor="tab:cyan",
                linewidth=2,
                label="Stimulus",
            ),
        )
        ax.legend(loc="upper right")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_aspect("equal")

    return fig, ax
