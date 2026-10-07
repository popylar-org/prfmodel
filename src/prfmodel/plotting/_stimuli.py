"""Stimuli plotting functions."""

from collections.abc import Sequence
from typing import Literal
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from prfmodel.stimuli import CSFStimulus
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import _EXPECTED_NDIM
from ._utils import _get_figure_axes

_MAX_TICK_LABELS = 20


def _setup_2d_plot(
    stimulus: PRFStimulus,
    title: str | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes, tuple[float, float, float, float]]:
    """Set up a 2D plot for a stimulus."""
    num_dim = stimulus.grid.shape[-1]

    if num_dim != _EXPECTED_NDIM:
        msg = f"Stimulus must be 2-dimensional, but has {num_dim} dimensions"
        raise ValueError(msg)

    grid_limits = _get_grid_limits(stimulus.grid)

    fig, ax = plt.subplots(**kwargs)

    if stimulus.dimension_labels:
        ax.set_ylabel(stimulus.dimension_labels[0])
        ax.set_xlabel(stimulus.dimension_labels[1])

    if title:
        ax.set_title(title)

    return fig, ax, grid_limits


def animate_2d_prf_stimulus(  # noqa: PLR0913
    stimulus: PRFStimulus,
    title: str | None = None,
    origin: Literal["upper", "lower"] = "lower",
    interval: int = 50,
    blit: bool = True,
    repeat_delay: int = 1000,
    **kwargs,
) -> animation.ArtistAnimation:
    """Animate a two-dimensional population receptive field stimulus.

    Parameters
    ----------
    stimulus : PRFStimulus
        The population receptive field stimulus to visualize.
    title : str or None, optional
        Title for the video animation.
    origin : str, optional
        `origin` argument for :meth:`matplotlib.axes.Axes.imshow`.
    interval : int, optional
        `interval` argument passed to :class:`matplotlib.animation.ArtistAnimation`.
    blit : bool, optional
        `blit` argument passed to :class:`matplotlib.animation.ArtistAnimation`.
    repeat_delay : int, optional
        `repeat_delay` argument passed to :class:`matplotlib.animation.ArtistAnimation`.
    kwargs
        Extra arguments passed to :func:`matplotlib.pyplot.subplots`
        and :class:`matplotlib.animation.ArtistAnimation`.

    Returns
    -------
    matplotlib.animation.ArtistAnimation

    Raises
    ------
    ValueError
        If the stimulus is not 2-dimensional.

    Notes
    -----
    The function uses matplotlib under the hood, and you can use the :data:`matplotlib.rcParams`
    to customize the animation, as described on the
    `matplotlib docs <https://matplotlib.org/stable/users/explain/customizing.html>`_.

    Examples
    --------
    >>> from IPython.display import HTML  # doctest: +SKIP
    >>> bar_stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=100, width=128, height=64)
    >>> ani = animate_2d_prf_stimulus(bar_stimulus)
    >>> video = ani.to_html5_video()  # doctest: +SKIP
    >>> HTML(video)  # doctest: +SKIP

    """
    fig, ax, grid_limits = _setup_2d_plot(stimulus, title, **kwargs)

    n_frames = stimulus.design.shape[0]
    ims = []
    for i in range(n_frames):
        im = ax.imshow(stimulus.design[i, :, :], animated=True, extent=grid_limits, origin=origin)
        ims.append([im])

    kwargs = kwargs | {"interval": interval, "blit": blit, "repeat_delay": repeat_delay}
    ani = animation.ArtistAnimation(fig, ims, **kwargs)
    plt.close(fig)
    return ani


def plot_2d_prf_stimulus(
    stimulus: PRFStimulus,
    frame_idx: int,
    origin: Literal["upper", "lower"] = "lower",
    title: str | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot a single frame of a two-dimensional population receptive field stimulus.

    Parameters
    ----------
    stimulus : PRFStimulus
        The population receptive field stimulus to visualize.
    frame_idx : int
        Index of the frame to plot.
    origin : str, optional
        `origin` argument for :meth:`matplotlib.axes.Axes.imshow`.
    title : str or None, optional
        Title for the plot.
    kwargs
        Extra arguments passed to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If the stimulus is not 2-dimensional.

    """
    fig, ax, grid_limits = _setup_2d_plot(stimulus, title, **kwargs)

    ax.imshow(stimulus.design[frame_idx, :, :], extent=grid_limits, origin=origin)

    plt.close(fig)
    return fig, ax


def plot_1d_prf_stimulus(  # noqa: PLR0913
    stimulus: PRFStimulus,
    tick_labels: Sequence[str | float] | None = None,
    secondary_tick_labels: Sequence[str | float] | None = None,
    ylabel: str | None = None,
    secondary_ylabel: str | None = None,
    origin: Literal["upper", "lower"] = "lower",
    ax: mpl.axes.Axes | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """Plot the design of a one-dimensional population receptive field stimulus.

    Shows the design as an image with time frames on the x-axis and the stimulus coordinates on the y-axis. Each row
    corresponds to one coordinate of the stimulus grid. The coordinates do not need to be regularly spaced.

    Parameters
    ----------
    stimulus : PRFStimulus
        The one-dimensional population receptive field stimulus to visualize.
    tick_labels : Sequence[str or float] or None, optional
        Labels for each coordinate on the y-axis. If `None` and the grid has at most 20 coordinates, the grid values
        (rounded to two decimals) are used. Otherwise, the y-axis shows the coordinate index.
    secondary_tick_labels : Sequence[str or float] or None, optional
        Labels for each coordinate on a secondary y-axis on the right (e.g., the coordinates on a different scale).
        If `None`, no secondary axis is drawn.
    ylabel : str or None, optional
        Label of the y-axis. If `None`, the dimension label of the stimulus is used (if available).
    secondary_ylabel : str or None, optional
        Label of the secondary y-axis.
    origin : str, optional
        `origin` argument for :meth:`matplotlib.axes.Axes.imshow`. With `"lower"`, the first coordinate is at the
        bottom.
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
        If the stimulus is not 1-dimensional or the number of tick labels does not match the number of coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> design = np.eye(5)[np.random.default_rng(0).integers(0, 5, size=50)]
    >>> grid = np.log(np.arange(1.0, 6.0))[:, np.newaxis]
    >>> stimulus = PRFStimulus(design=design, grid=grid, dimension_labels=["log(numerosity)"])
    >>> fig, ax = plot_1d_prf_stimulus(stimulus, secondary_tick_labels=[1, 2, 3, 4, 5])

    """
    num_dim = stimulus.grid.shape[-1]

    if num_dim != 1:
        msg = f"Stimulus must be 1-dimensional, but has {num_dim} dimensions"
        raise ValueError(msg)

    coordinates = stimulus.grid[..., 0]
    num_coordinates = coordinates.shape[0]

    if tick_labels is None and num_coordinates <= _MAX_TICK_LABELS:
        tick_labels = [f"{value:.2f}" for value in coordinates]

    for labels in (tick_labels, secondary_tick_labels):
        if labels is not None and len(labels) != num_coordinates:
            msg = f"Number of tick labels ({len(labels)}) must match the number of coordinates ({num_coordinates})"
            raise ValueError(msg)

    if ylabel is None and stimulus.dimension_labels:
        ylabel = stimulus.dimension_labels[0]

    fig, ax = _get_figure_axes(ax, **kwargs)

    ax.imshow(stimulus.design.T, aspect="auto", interpolation="none", origin=origin)

    ax.set_xlabel("Time frame")

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ticks = np.arange(num_coordinates)

    if tick_labels is not None:
        ax.set_yticks(ticks, labels=[str(label) for label in tick_labels])

    if secondary_tick_labels is not None:
        secax = ax.secondary_yaxis("right")
        secax.set_yticks(ticks, labels=[str(label) for label in secondary_tick_labels])

        if secondary_ylabel is not None:
            secax.set_ylabel(secondary_ylabel)

    return fig, ax


def _get_grid_limits(grid: np.ndarray) -> tuple[float, float, float, float]:
    """From a 2D coordinate grid, return its coordinate limits.

    The grid stores the vertical (`y`) coordinate in `grid[..., 0]` and the horizontal (`x`)
    coordinate in `grid[..., 1]`, matching the row-major order of the design (see
    :class:`~prfmodel.stimuli.PRFStimulus`). Rows therefore span `y` and columns span `x`.

    Output can be passed as `extent` argument to :class:`matlplotlib.axes.Axes.imshow`

    """
    # x varies along columns (axis 1) and is stored in the last grid dimension
    left = grid[0, 0, 1]
    right = grid[0, -1, 1]

    # y varies along rows (axis 0) and is stored in the first grid dimension
    bottom = grid[0, 0, 0]
    top = grid[-1, 0, 0]

    return (left, right, bottom, top)


def plot_csf_stimulus_design(stimulus: CSFStimulus, **kwargs) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot the design of a contrast sensitivity function (CSF) stimulus.

    Plots the stimulus contrast over time with color indicating different spatial frequencies.

    Parameters
    ----------
    stimulus : CSFStimulus
        Stimulus object with contrast and spatial frequencies.
    **kwargs:
        Keyword arguments passed to :func:`~matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    See Also
    --------
    plot_csf_stimulus_curve : Contrast sensitivity curve on top of the stimulus design.

    """
    sf_levels = np.unique(stimulus.sf)

    fig, ax = plt.subplots(**kwargs)

    for freq in sf_levels:
        is_freq = stimulus.sf == freq
        con = stimulus.contrast[is_freq]
        frames = np.argmax(is_freq) + np.arange(con.shape[0])
        ax.plot(frames, con, label=freq)

    ax.set_xlabel("Time frame")
    ax.set_ylabel("Contrast")

    fig.legend(title="Spatial\nfrequency", bbox_to_anchor=(1.05, 1))

    return fig, ax


def plot_csf_stimulus_curve(
    stimulus: CSFStimulus,
    csf: np.ndarray,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    Plot a contrast sensitivity curve over a stimulus design.

    Plots spatial frequency over contrast sensitivity (100/contrast) and draws a contrast sentitivity curve on top.

    Parameters
    ----------
    stimulus : CSFStimulus
        Stimulus object with contrast and spatial frequencies.
    csf : numpy.ndarray
        Contrast sensitivity curve. Must match the number of frames in ``stimulus``.
    **kwargs:
        Keyword arguments passed to :func:`~matplotlib.pyplot.subplots`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.

    See Also
    --------
    plot_csf_stimulus_design : Stimulus contrast over time.

    """
    csf = np.atleast_2d(csf)

    if csf.shape[1] != stimulus.contrast.shape[0]:
        msg = f"'csf' must have the same number of frames as 'stimulus' but has only {csf.shape[1]} frames"
        raise ValueError(msg)

    sf_levels = np.unique(stimulus.sf)
    fig, ax = plt.subplots(**kwargs)

    for freq in sf_levels:
        is_freq = stimulus.sf == freq
        con = stimulus.contrast[is_freq]
        ax.scatter(stimulus.sf[is_freq], 100.0 / con, label=freq)

    ax.loglog(stimulus.sf, np.transpose(csf), color="black")

    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("Contrast sensitivity")

    return fig, ax
