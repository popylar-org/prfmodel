"""Stimuli plotting functions."""

from typing import Literal
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from prfmodel.stimuli import PRFStimulus
from prfmodel.utils import _EXPECTED_NDIM


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


def _get_grid_limits(grid: np.ndarray) -> tuple[float, float, float, float]:
    """From a 2D coordinate grid, return its coordinate limits.

    Output can be passed as `extent` argument to :class:`matlplotlib.axes.Axes.imshow`

    """
    left = grid[0, 0, 0]
    bottom = grid[0, 0, -1]

    right = grid[0, -1, 0]
    top = grid[-1, -1, -1]
    return (left, right, bottom, top)
