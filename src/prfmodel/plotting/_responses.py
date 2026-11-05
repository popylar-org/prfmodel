"""Response plotting functions."""

from collections.abc import Mapping
import matplotlib as mpl
import numpy as np
from prfmodel.utils import _EXPECTED_NDIM
from ._utils import _get_figure_axes


def plot_response_heatmap(  # noqa: PLR0913
    response: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "inferno",
    xlabel: str = "Time frame",
    ylabel: str = "Unit index",
    colorbar_label: str | None = "Response",
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the response of many units over time as a heatmap.

    Gives an overview of all observed or predicted timecourses at once (also known as a carpet plot). Each row shows
    the response of one unit (e.g., voxel or vertex) and each column one time frame.

    Parameters
    ----------
    response : numpy.ndarray
        Response with shape `(num_units, num_frames)`.
    vmin : float or None, optional
        Lower limit of the color scale.
    vmax : float or None, optional
        Upper limit of the color scale.
    cmap : str, optional
        Name of the colormap.
    xlabel : str, optional
        Label of the x-axis.
    ylabel : str, optional
        Label of the y-axis.
    colorbar_label : str or None, optional
        Label of the colorbar. If `None`, no colorbar is drawn.
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
        If `response` is not 2-dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> response = np.random.default_rng(0).normal(size=(500, 120))
    >>> fig, ax = plot_response_heatmap(response, vmin=-2.0, vmax=2.0)

    """
    response = np.asarray(response)

    if response.ndim != _EXPECTED_NDIM:
        msg = f"'response' must be 2-dimensional, but has {response.ndim} dimensions"
        raise ValueError(msg)

    fig, ax = _get_figure_axes(ax, **kwargs)

    im = ax.imshow(response, aspect="auto", interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if colorbar_label is not None:
        fig.colorbar(im, ax=ax, label=colorbar_label)

    return fig, ax


def plot_observed_predicted(  # noqa: PLR0913
    observed: np.ndarray,
    predicted: np.ndarray | Mapping[str, np.ndarray],
    split: int | None = None,
    observed_label: str = "Observed",
    xlabel: str = "Time frame",
    ylabel: str = "Response",
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot an observed timecourse together with one or more predicted timecourses.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed (or simulated) timecourse of a single unit with shape `(num_frames,)`.
    predicted : numpy.ndarray or Mapping[str, numpy.ndarray]
        Predicted timecourse with shape `(num_frames,)`. To compare several predictions (e.g., from different fitters),
        pass a mapping from labels to predicted timecourses. Predictions are drawn as dashed lines.
    split : int or None, optional
        Time frame at which to draw a vertical line (e.g., the split between a training and test set).
    observed_label : str, optional
        Legend label of the observed timecourse.
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
        If a predicted timecourse does not have the same shape as the observed timecourse.

    Examples
    --------
    >>> import numpy as np
    >>> observed = np.sin(np.linspace(0.0, 10.0, 100))
    >>> fig, ax = plot_observed_predicted(observed, {"Grid": 0.8 * observed, "SGD": 0.95 * observed})

    """
    observed = np.squeeze(np.asarray(observed))

    if not isinstance(predicted, Mapping):
        predicted = {"Predicted": predicted}

    fig, ax = _get_figure_axes(ax, **kwargs)

    ax.plot(observed, label=observed_label)

    for label, pred in predicted.items():
        pred_squeezed = np.squeeze(np.asarray(pred))

        if pred_squeezed.shape != observed.shape:
            msg = (
                f"Predicted timecourse '{label}' must have shape {observed.shape} like the observed timecourse, "
                f"but has shape {pred_squeezed.shape}"
            )
            raise ValueError(msg)

        ax.plot(pred_squeezed, "--", label=label)

    if split is not None:
        ax.axvline(split, color="gray", linestyle=":")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    return fig, ax
