"""Model evaluation plotting functions."""

from collections.abc import Mapping
import matplotlib as mpl
import numpy as np
from ._utils import _get_figure_axes


def plot_r_squared_hist(  # noqa: PLR0913
    r_squared: np.ndarray | Mapping[str, np.ndarray],
    bins: int = 40,
    value_range: tuple[float, float] = (-0.5, 1.0),
    clip: bool = False,
    log: bool = False,
    xlabel: str = "R-squared",
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the distribution of R-squared scores across units.

    Parameters
    ----------
    r_squared : numpy.ndarray or Mapping[str, numpy.ndarray]
        R-squared score of each unit with shape `(num_units,)`. To compare several sets of scores (e.g., in-sample and
        out-of-sample), pass a mapping from labels to scores. The histograms are then overlaid and labeled in a legend.
        `NaN` values are ignored.
    bins : int, optional
        Number of histogram bins.
    value_range : tuple[float, float], optional
        Lower and upper limit of the histogram bins.
    clip : bool, optional
        Whether to clip scores to `value_range` so that scores outside the range are counted in the outermost bins.
        If `False`, scores outside the range are not shown.
    log : bool, optional
        Whether to use a logarithmic scale for the counts. Useful when most units have a score close to zero.
    xlabel : str, optional
        Label of the x-axis.
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

    See Also
    --------
    plot_r_squared_comparison : Compare two sets of R-squared scores unit by unit.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r_squared = {"Training set": rng.uniform(size=100), "Test set": rng.uniform(size=100)}
    >>> fig, ax = plot_r_squared_hist(r_squared, log=True)

    """
    is_mapping = isinstance(r_squared, Mapping)
    scores: dict[str | None, np.ndarray]

    if isinstance(r_squared, Mapping):
        scores = {label: values for label, values in r_squared.items()}  # noqa: C416 (dict() is not type-compatible)
    else:
        scores = {None: r_squared}  # A single set of scores has no label

    fig, ax = _get_figure_axes(ax, **kwargs)

    bin_edges = np.linspace(*value_range, bins + 1).tolist()
    # Overlaid histograms need to be transparent to remain visible
    alpha = 0.5 if len(scores) > 1 else 1.0

    for label, values in scores.items():
        values_array = np.asarray(values, dtype=float).ravel()
        values_array = values_array[~np.isnan(values_array)]

        if clip:
            values_array = np.clip(values_array, *value_range)

        ax.hist(values_array, bins=bin_edges, alpha=alpha, label=label)

    if log:
        ax.set_yscale("log")

    ax.set_xlim(*value_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of units")

    if is_mapping:
        ax.legend()

    return fig, ax


def plot_r_squared_comparison(  # noqa: PLR0913
    r_squared_x: np.ndarray,
    r_squared_y: np.ndarray,
    value_range: tuple[float, float] = (-0.5, 1.0),
    gridsize: int = 50,
    cmap: str = "inferno",
    xlabel: str = "R-squared (training set)",
    ylabel: str = "R-squared (test set)",
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Compare two sets of R-squared scores unit by unit.

    Plots a two-dimensional histogram (with hexagonal bins and logarithmic counts) of two scores for each unit
    together with the identity line. Useful to compare in-sample with out-of-sample scores or scores of two models.
    Units above the identity line have a higher score on the y-axis.

    Parameters
    ----------
    r_squared_x : numpy.ndarray
        R-squared score of each unit on the x-axis with shape `(num_units,)`.
    r_squared_y : numpy.ndarray
        R-squared score of each unit on the y-axis with shape `(num_units,)`.
    value_range : tuple[float, float], optional
        Lower and upper limit of both axes.
    gridsize : int, optional
        Number of hexagons in the x-direction.
    cmap : str, optional
        Name of the colormap.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str, optional
        Label of the y-axis.
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
        If `r_squared_x` and `r_squared_y` do not have the same shape.

    See Also
    --------
    plot_r_squared_hist : Distribution of R-squared scores across units.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> r_squared_train = rng.uniform(size=100)
    >>> r_squared_test = r_squared_train - rng.uniform(0.0, 0.2, size=100)
    >>> fig, ax = plot_r_squared_comparison(r_squared_train, r_squared_test)

    """
    r_squared_x = np.asarray(r_squared_x, dtype=float).ravel()
    r_squared_y = np.asarray(r_squared_y, dtype=float).ravel()

    if r_squared_x.shape != r_squared_y.shape:
        msg = (
            f"'r_squared_x' and 'r_squared_y' must have the same shape, but have shapes {r_squared_x.shape} "
            f"and {r_squared_y.shape}"
        )
        raise ValueError(msg)

    fig, ax = _get_figure_axes(ax, **kwargs)

    is_valid = ~(np.isnan(r_squared_x) | np.isnan(r_squared_y))

    hexbin = ax.hexbin(
        r_squared_x[is_valid],
        r_squared_y[is_valid],
        gridsize=gridsize,
        bins="log",
        mincnt=1,
        cmap=cmap,
        extent=(*value_range, *value_range),
    )
    ax.plot(value_range, value_range, color="gray", linestyle="--")  # Identity line

    fig.colorbar(hexbin, ax=ax, label="Number of units")

    ax.set_xlim(*value_range)
    ax.set_ylim(*value_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    return fig, ax
