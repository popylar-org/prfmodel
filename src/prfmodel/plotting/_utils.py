"""Shared plotting helpers."""

from collections.abc import Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Two values are always both regularly and log-spaced
_MIN_LOG_DETECT_SIZE = 2


def _get_figure_axes(ax: mpl.axes.Axes | None, **kwargs) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Return the figure of an existing axes or create a new figure and axes."""
    if ax is None:
        return plt.subplots(**kwargs)

    fig = ax.get_figure()

    # Go up to the root figure if the axes are in a subfigure
    while isinstance(fig, mpl.figure.SubFigure):
        fig = fig.figure

    if not isinstance(fig, mpl.figure.Figure):
        msg = "'ax' must belong to a figure"
        raise TypeError(msg)

    return fig, ax


def _is_regular(values: np.ndarray) -> bool:
    """Check whether sorted values are regularly spaced."""
    steps = np.diff(values)
    return bool(np.allclose(steps, steps[0]))


def _is_log_spaced(values: np.ndarray | Sequence[float]) -> bool:
    """Check whether grid values are log-spaced but not regularly spaced."""
    values = np.sort(np.asarray(values, dtype=float))

    if values.size <= _MIN_LOG_DETECT_SIZE or np.any(values <= 0.0) or _is_regular(values):
        return False

    return _is_regular(np.log(values))


def _grid_bin_edges(values: np.ndarray | Sequence[float], log: bool = False) -> np.ndarray:
    """Compute histogram bin edges centered on the values of a regularly (or log-) spaced grid."""
    values = np.sort(np.asarray(values, dtype=float))

    if values.size == 1:
        msg = "Grid must contain at least two values to compute bin edges"
        raise ValueError(msg)

    if log:
        if np.any(values <= 0.0):
            msg = "Grid values must be positive when 'log=True'"
            raise ValueError(msg)
        values = np.log(values)

    steps = np.diff(values)

    if not np.allclose(steps, steps[0]):
        spacing = "log-spaced" if log else "regularly spaced"
        msg = f"Grid values must be {spacing}"
        raise ValueError(msg)

    edges = np.append(values - steps[0] / 2, values[-1] + steps[0] / 2)

    return np.exp(edges) if log else edges
