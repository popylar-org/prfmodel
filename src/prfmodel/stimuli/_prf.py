"""Container for population receptive field stimulus design and grid."""

from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
from keras import ops
from prfmodel.exceptions import ShapeError
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .base import Stimulus
from .base import StimulusTensors


@dataclass(frozen=True, eq=False)
class PRFStimulusTensors(StimulusTensors):
    """Tensor-holding counterpart of a :class:`~prfmodel.stimuli.PRFStimulus`.

    Holds the population receptive field stimulus arrays as backend tensors. Should be created with
    :meth:`PRFStimulus.to_tensors`.

    Parameters
    ----------
    design : :data:`prfmodel.typing.Tensor`
        The :attr:`PRFStimulus.design` array as a tensor.
    grid : :data:`prfmodel.typing.Tensor`
        The :attr:`PRFStimulus.grid` array as a tensor.

    """

    design: Tensor
    grid: Tensor


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

        The order of the last axis must follow the array axes of the design: `grid[..., i]` holds the
        coordinate that varies along design axis `i + 1`. For a 2D stimulus with a design of shape
        `(num_frames, height, width)` this means `grid[..., 0]` is the vertical (`y`) coordinate and
        `grid[..., 1]` is the horizontal (`x`) coordinate. See Notes.
    dimension_labels : Sequence[str] or None, optional
        Names of the grid dimensions (e.g., `["y", "x"]`). If given, the number of labels must match the last grid
        axis. Label `i` names the coordinate stored in `grid[..., i]`.

    Raises
    ------
    ShapeError
        If the number of grid axes (excluding the last) does not match the last grid dimension.
    ShapeMismatchError
        If the design and grid dimensions do not match.
    ValueError
        If the number of dimension labels does not match the last grid dimension.

    Notes
    -----
    The shapes of the design and grid must match according to `design.shape[1:] == grid.shape[:-1]`.
    That is, all design dimensions but the first must have the same size as the grid
    dimensions excluding the last grid dimension.

    **Coordinate order.** The last grid axis is ordered to match the design's array axes, so
    `grid[..., i]` is the coordinate that varies along design axis `i + 1`. Because NumPy uses
    row-major order, the first design axis after time is rows (height), which means the vertical
    (`y`) coordinate comes first and the horizontal (`x`) coordinate second.

    This is why response models read their centre parameters in the same order, for example
    :class:`~prfmodel.models.prf.Gaussian2DPRFResponse` pairs `mu_y` with `grid[..., 0]` and `mu_x`
    with `grid[..., 1]`. Building a grid with `numpy.meshgrid` therefore requires stacking `yv`
    before `xv`, as in the example below.

    Examples
    --------
    Create a population receptive field stimulus on a 2D grid.

    >>> import numpy as np
    >>> # Deliberately different height and width so the axis order is unambiguous
    >>> num_frames, height, width = 10, 8, 16
    >>> design = np.ones((num_frames, height, width))  # time first, then rows, then columns
    >>> pixel_size = 0.5
    >>> x = (np.arange(width) - (width - 1) / 2) * pixel_size
    >>> y = (np.arange(height) - (height - 1) / 2) * pixel_size
    >>> xv, yv = np.meshgrid(x, y)
    >>> # y comes first because design axis 1 is the row (height) axis
    >>> grid = np.stack((yv, xv), axis=-1)  # shape (height, width, 2)
    >>> # The (y, x) coordinates of the bottom-left corner:
    >>> grid[0, 0, :]
    array([-1.75, -3.75])
    >>> # The (y, x) coordinates of the top-right corner:
    >>> grid[7, 15, :]
    array([1.75, 3.75])
    >>> # Moving along the last column changes x only:
    >>> grid[0, 15, :]
    array([-1.75,  3.75])
    >>> stimulus = PRFStimulus(design=design, grid=grid, dimension_labels=["y", "x"])
    >>> print(stimulus)
    PRFStimulus(design=array[10, 8, 16], grid=array[8, 16, 2], dimension_labels=['y', 'x'])

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
            raise ShapeMismatchError("design", self.design.shape, "grid", self.grid.shape)  # noqa: EM101 (exception literal)

    def _check_grid_dimensions(self) -> None:
        num_grid_axes = len(self.grid.shape[:-1])
        last_dim = self.grid.shape[-1]
        if num_grid_axes != last_dim:
            raise ShapeError("grid", self.grid.shape, f"must have axes matching the last dimension {last_dim}")  # noqa: EM101 (exception literal)

    def _check_dimension_labels(self) -> None:
        if self.dimension_labels is not None and self.grid.shape[-1] != len(self.dimension_labels):
            msg = (
                f"Length of 'dimension_labels' {len(self.dimension_labels)} does not match "
                f"last grid dimension {self.grid.shape[-1]}"
            )
            raise ValueError(msg)

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

        Notes
        -----
        The `design` is stored in **screen pixel** order, but the `grid` holds **visual field**
        coordinates in degrees of visual angle. These two spaces are related by a horizontal flip,
        because in a typical MRI setup the participant views the screen through a mirror that
        reverses left and right. The horizontal coordinate therefore *decreases* across the design
        columns: `grid[0, 0, 1]` is the largest (most positive) `x` and `grid[0, -1, 1]` the
        smallest. The vertical coordinate is unaffected and increases down the rows.

        Concretely, a bar drawn in the leftmost columns of `design` falls in the **right** visual
        hemifield (positive `mu_x`), and a bar in the rightmost columns falls in the left hemifield
        (negative `mu_x`). A ``direction="horizontal"`` bar sweeps left-to-right across the screen,
        which is right-to-left (decreasing `mu_x`) through the visual field.

        This is the same convention as the packaged stimulus returned by
        :func:`~prfmodel.examples.load_2d_prf_bar_stimulus`, so `mu_x` estimates are directly
        comparable between the two.

        Examples
        --------
        >>> stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=200)
        >>> print(stimulus)
        PRFStimulus(design=array[200, 128, 128], grid=array[128, 128, 2], dimension_labels=['y', 'x'])

        The horizontal coordinate decreases across the columns, the vertical one increases down the
        rows:

        >>> stimulus = PRFStimulus.create_2d_bar_stimulus(width=5, height=3, pixel_size=1.0)
        >>> stimulus.grid[..., 1]  # x, constant down each column
        array([[ 2.,  1.,  0., -1., -2.],
               [ 2.,  1.,  0., -1., -2.],
               [ 2.,  1.,  0., -1., -2.]])
        >>> stimulus.grid[..., 0]  # y, constant across each row
        array([[-1., -1., -1., -1., -1.],
               [ 0.,  0.,  0.,  0.,  0.],
               [ 1.,  1.,  1.,  1.,  1.]])

        """
        # Create a centered grid of x and y coordinates.
        # x decreases across the columns: the design is stored in screen pixel order while the grid
        # holds visual field coordinates, and the mirror in the scanner flips the horizontal axis
        # (see Notes). This matches the packaged stimulus from
        # :func:`~prfmodel.examples.load_2d_prf_bar_stimulus`.
        x = ((width - 1) / 2 - np.arange(width)) * pixel_size
        y = (np.arange(height) - (height - 1) / 2) * pixel_size
        xv, yv = np.meshgrid(x, y)
        # Dimension y comes first because numpy uses row-major order (i.e., the first design axis
        # after time represents rows or height), so grid[..., 0] must vary along that axis
        grid = np.stack((yv, xv), axis=-1)  # shape (height, width, 2)

        # Create the design array. Positions below are screen pixel columns/rows, not visual field
        # coordinates; the horizontal axis of the two is reversed (see Notes).
        design = np.zeros((num_frames, height, width), dtype=np.float32)

        for frame in range(num_frames):
            if direction == "horizontal":
                # Bar moves left to right across the screen (right to left through the visual
                # field), starting and ending just outside the screen
                bar_start = int(np.round(-bar_width + frame * (width + bar_width) / (num_frames - 1)))
                bar_end = bar_start + bar_width
                # Only draw within screen bounds
                screen_start = max(bar_start, 0)
                screen_end = min(bar_end, width)

                if screen_start < screen_end:
                    design[frame, :, screen_start:screen_end] = 1.0
            elif direction == "vertical":
                # Bar moves across the rows in index order (upward through the visual field, since
                # y increases down the rows), starting and ending just outside the screen
                bar_start = int(np.round(-bar_width + frame * (height + bar_width) / (num_frames - 1)))
                bar_end = bar_start + bar_width
                screen_start = max(bar_start, 0)
                screen_end = min(bar_end, height)

                if screen_start < screen_end:
                    design[frame, screen_start:screen_end, :] = 1.0
            else:
                msg = "Direction must be 'horizontal' or 'vertical'"
                raise ValueError(msg)

        dimension_labels = ["y", "x"]

        return cls(
            design=design,
            grid=grid,
            dimension_labels=dimension_labels,
        )

    def to_tensors(self, dtype: str | None = None) -> PRFStimulusTensors:
        """Convert the stimulus arrays into backend tensors.

        Parameters
        ----------
        dtype : str, optional
            The dtype to convert the stimulus arrays to. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        PRFStimulusTensors
            The stimulus arrays as tensors.

        Examples
        --------
        >>> from prfmodel.examples import load_2d_prf_bar_stimulus
        >>> stimulus = load_2d_prf_bar_stimulus()
        >>> tensors = stimulus.to_tensors("float32")
        >>> print(tuple(tensors.design.shape) == stimulus.design.shape)
        True
        >>> print(tuple(tensors.grid.shape) == stimulus.grid.shape)
        True

        """
        dtype = get_dtype(dtype)

        return PRFStimulusTensors(
            design=ops.convert_to_tensor(self.design, dtype=dtype),
            grid=ops.convert_to_tensor(self.grid, dtype=dtype),
        )
