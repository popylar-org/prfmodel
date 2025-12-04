"""Container for population receptive field stimulus design and grid."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from .base import Stimulus


class GridDesignShapeError(Exception):
    """
    Exception raised when the shapes of the design and grid do not match.

    Parameters
    ----------
    design_shape : tuple of int
        Shape of the design array.
    grid_shape : tuple of int
        Shape of the grid array.

    """

    def __init__(self, design_shape: tuple[int, ...], grid_shape: tuple[int, ...]):
        super().__init__(f"Shapes of 'design' {design_shape} and 'grid' {grid_shape} do not match")


class GridDimensionsError(Exception):
    """
    Exception raised when number of grid dimensions except for the last does not match last grid dimension size.

    Parameters
    ----------
    grid_shape: tuple of int
        Shape of the grid array.

    """

    def __init__(self, grid_shape: tuple[int, ...]) -> None:
        num_grid_axes = len(grid_shape[:-1])
        super().__init__(
            f"The number of dimensions in 'grid' {num_grid_axes} does not match its last dimension {grid_shape[-1]}",
        )


class DimensionLabelsError(Exception):
    """
    Exception raised when the number of dimensions does not match the grid's last dimension.

    Parameters
    ----------
    dimensions_len : int
        Length of the dimensions sequence.
    grid_dim : int
        Size of the last dimension of the grid.

    """

    def __init__(self, dimensions_len: int, grid_dim: int):
        super().__init__(f"Length of 'dimensions' {dimensions_len} does not match last dimension of 'grid' {grid_dim}")


class StimulusDimensionError(Exception):
    """Exception raised when Stimulus does not have the right number of dimensions.

    The dimension for the frames is ignored.

    Parameters
    ----------
    actual : int
        Number of dimensions in the stimulus grid.
    expected : int
        Number of expected dimensions in the stimulus grid.

    """

    def __init__(self, actual: int, expected: int):
        super().__init__(f"Stimulus frames have {actual} dimensions, but expected {expected}.")


@dataclass(frozen=True, eq=False)
class PRFStimulus(Stimulus):
    """
    Container for a population receptive field stimulus design and its associated grid.

    Parameters
    ----------
    design : numpy.ndarray
        The stimulus design array containing the stimulus value in one or more dimensions over different time frames.
        The first axis is assumed to be time frames. Additional axes represent design dimensions.
    grid : numpy.ndarray
        The coordinate system of the design. The last axis is the number of design dimensions
        excluding the time frame dimension. The shape excluding the last axis must match the shape
        of the design excluding the first axis.
    dimension_labels : Sequence[str] or None, optional
        Names of the grid dimensions (e.g., `["y", "x"]`). If given, the number of labels must match the last grid axis.

    Raises
    ------
    GridDesignShapeError
        If the design and grid dimensions do not match.
    GridDimensionsError
        If the number of dimensions of the grid except the last does not match the size of the last grid dimension.
    DimensionLabelsError
        If the number of dimensions does not match the grid's last dimension.

    Notes
    -----
    The shapes of the design and grid must match according to `design.shape[1:] == grid.shape[:-1]`.
    That is, all design dimensions but the first must have the same size as the grid
    dimensions excluding the last grid dimension.

    Examples
    --------
    Create a population receptive field stimulus on a 2D grid.

    >>> import numpy as np
    >>> num_frames, width, height = 10, 16, 16
    >>> design = np.ones((num_frames, width, height))
    >>> pixel_size = 0.05
    >>> x = (np.arange(width) - (width - 1) / 2) * pixel_size
    >>> y = (np.arange(height) - (height - 1) / 2) * pixel_size
    >>> xv, yv = np.meshgrid(x, y)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (height, width, 2)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (height, width, 2)
    >>> # The coordinates of the bottom-left corner:
    >>> grid[0, 0, :]
    array([-0.375, -0.375])
    >>> # The coordinates of the top-right corner:
    >>> grid[15, 15, :]
    array([0.375, 0.375])
    >>> stimulus = PRFStimulus(design=design, grid=grid, dimension_labels=["y", "x"])
    >>> print(stimulus)
    PRFStimulus(design=array[10, 16, 16], grid=array[16, 16, 2], dimension_labels=['y', 'x'])

    """

    design: np.ndarray
    grid: np.ndarray
    dimension_labels: Sequence[str] | None = None

    def __post_init__(self):
        self._check_grid_design_shape()
        self._check_grid_dimensions()
        self._check_dimension_labels()

    def _check_grid_design_shape(self) -> None:
        if not self.design.shape[1:] == self.grid.shape[:-1]:
            raise GridDesignShapeError(self.design.shape, self.grid.shape)

    def _check_grid_dimensions(self) -> None:
        if not len(self.grid.shape[:-1]) == self.grid.shape[-1]:
            raise GridDimensionsError(self.grid.shape)

    def _check_dimension_labels(self) -> None:
        if self.dimension_labels is not None and not self.grid.shape[-1] == len(self.dimension_labels):
            raise DimensionLabelsError(len(self.dimension_labels), self.grid.shape[-1])

    @classmethod
    def create_2d_bar_stimulus(  # noqa: PLR0913 (too many arguments)
        cls,
        num_frames: int = 100,
        width: int = 128,
        height: int = 128,
        bar_width: int = 20,
        direction: str = "horizontal",
        pixel_size: float = 0.05,
    ) -> "PRFStimulus":
        """
        Create a population receptive field bar stimulus that moves across a 2D screen.

        The stimulus starts and ends moving just outside the screen.

        Parameters
        ----------
        num_frames : int, optional
            Number of time frames in the stimulus.
        width : int, optional
            Width of the stimulus grid (in pixels).
        height : int, optional
            Height of the stimulus grid (in pixels).
        bar_width : int, optional
            Width of the moving bar (in pixels).
        direction : {"horizontal", "vertical"}, optional
            Direction in which the bar moves.
        pixel_size : float, optional
            Size of a pixel in spatial units.

        Returns
        -------
        PRFStimulus
            A stimulus instance with the generated design and grid.

        Raises
        ------
        ValueError
            If `direction` is not "horizontal" or "vertical".

        Examples
        --------
        >>> stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=200)
        >>> print(stimulus)
        PRFStimulus(design=array[200, 128, 128], grid=array[128, 128, 2], dimension_labels=['y', 'x'])

        """
        # Create a centered grid of x and y coordinates
        x = (np.arange(width) - (width - 1) / 2) * pixel_size
        y = (np.arange(height) - (height - 1) / 2) * pixel_size
        xv, yv = np.meshgrid(x, y)
        grid = np.stack((xv, yv), axis=-1)  # shape (height, width, 2)

        # Create the design array
        design = np.zeros((num_frames, height, width), dtype=np.float32)

        for frame in range(num_frames):
            if direction == "horizontal":
                # Bar moves left to right, starting and ending just outside the screen
                bar_start = int(np.round(-bar_width + frame * (width + bar_width) / (num_frames - 1)))
                bar_end = bar_start + bar_width
                # Only draw within screen bounds
                screen_start = max(bar_start, 0)
                screen_end = min(bar_end, width)

                if screen_start < screen_end:
                    design[frame, :, screen_start:screen_end] = 1.0
            elif direction == "vertical":
                # Bar moves top to bottom, starting and ending just outside the screen
                bar_start = int(np.round(-bar_width + frame * (height + bar_width) / (num_frames - 1)))
                bar_end = bar_start + bar_width
                screen_start = max(bar_start, 0)
                screen_end = min(bar_end, height)

                if screen_start < screen_end:
                    design[frame, screen_start:screen_end, :] = 1.0
            else:
                msg = "Direction must be 'horizontal' or 'vertical'"
                raise ValueError(msg)

        # Dimension y comes first because numpy uses row-major order (i.e., the first axis represents rows or height)
        dimension_labels = ["y", "x"]

        return cls(
            design=design,
            grid=grid,
            dimension_labels=dimension_labels,
        )


def _setup_2d_plot(
    stimulus: PRFStimulus,
    title: str | None = None,
    **kwargs,
) -> tuple[mpl.figure.Figure, mpl.axes.Axes, tuple[float, float, float, float]]:
    """Set up a 2D plot for a stimulus."""
    _verify_dimensions(stimulus, 2)

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
    StimulusDimensionError
        If `stimulus` is not 2-dimensional.

    Notes
    -----
    The function uses matplotlib under the hood, and you can use the :data:`matplotlib.rcParams`
    to customize the animation, as described on the
    `matplotlib docs <https://matplotlib.org/stable/users/explain/customizing.html>`_.

    Examples
    --------
    >>> from IPython.display import HTML
    >>> from prfmodel.stimulus import PRFStimulus, animate_2d_prf_stimulus
    >>> bar_stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=100, width=128, height=64)
    >>> ani = animate_2d_prf_stimulus(bar_stimulus)
    >>> video = ani.to_html5_video()
    >>> HTML(video)

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
    StimulusDimensionError
        If the stimulus is not 2-dimensional.

    """
    fig, ax, grid_limits = _setup_2d_plot(stimulus, title, **kwargs)

    ax.imshow(stimulus.design[frame_idx, :, :], extent=grid_limits, origin=origin)

    plt.close(fig)
    return fig, ax


def _verify_dimensions(stimulus: PRFStimulus, expected: int) -> None:
    """Verify that stimulus has the right dimensions.

    This checks for the number of dimensions excluding the first (frame) dimension.

    """
    actual = len(stimulus.design.shape)
    if actual != expected + 1:
        raise StimulusDimensionError(actual, expected)


def _get_grid_limits(grid: np.ndarray) -> tuple[float, float, float, float]:
    """From a 2D coordinate grid, return its coordinate limits.

    Output can be passed as `extent` argument to :class:`matlplotlib.axes.Axes.imshow`

    """
    left = grid[0, 0, 0]
    bottom = grid[0, 0, -1]

    right = grid[0, -1, 0]
    top = grid[-1, -1, -1]
    return (left, right, bottom, top)
