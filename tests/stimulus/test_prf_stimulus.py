"""Test PRFStimulus class."""

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import animation

# Needs to be imported to recreate stimulus from repr
from numpy import array  # noqa: F401
from prfmodel.stimulus.prf import DimensionLabelsError
from prfmodel.stimulus.prf import GridDesignShapeError
from prfmodel.stimulus.prf import GridDimensionsError
from prfmodel.stimulus.prf import PRFStimulus
from prfmodel.stimulus.prf import StimulusDimensionError
from prfmodel.stimulus.prf import _get_grid_limits
from prfmodel.stimulus.prf import _verify_dimensions
from prfmodel.stimulus.prf import animate_2d_prf_stimulus
from prfmodel.stimulus.prf import plot_2d_prf_stimulus


def test_grid_design_shape_error():
    """Check that shape mismatches are detected correctly by PRFStimulus."""
    with pytest.raises(GridDesignShapeError):
        _ = PRFStimulus(
            design=np.zeros((1, 2)),
            grid=np.zeros((1, 1)),
        )


def test_grid_dimension_error():
    """Check that shape mismatches are detected correctly by PRFStimulus."""
    with pytest.raises(GridDimensionsError):
        _ = PRFStimulus(
            design=np.zeros((1, 1, 2)),
            grid=np.zeros((1, 2, 1)),
        )


def test_dimension_labels_error():
    """Check that dimension mismatches are detected correctly by PRFStimulus."""
    with pytest.raises(DimensionLabelsError):
        _ = PRFStimulus(
            design=np.zeros((1, 2)),
            grid=np.zeros((2, 1)),
            dimension_labels=["x", "y"],
        )


@pytest.fixture
def stimulus():
    """Stimulus object."""
    return PRFStimulus(
        design=np.zeros((1, 2, 1)),
        grid=np.zeros((2, 1, 2)),
        dimension_labels=["x", "y"],
    )


@pytest.fixture
def stimulus_1d():
    """Stimulus object."""
    return PRFStimulus(
        design=np.zeros((1, 2)),
        grid=np.zeros((2, 1)),
        dimension_labels=["x"],
    )


def test_repr(stimulus: PRFStimulus):
    """Test machine-readable string representation of PRFStimulus."""
    stimulus_2 = eval(repr(stimulus))  # noqa: S307

    assert stimulus == stimulus_2


def test_str(stimulus: PRFStimulus):
    """Test human-readable string representation of PRFStimulus."""
    assert str(stimulus) == "PRFStimulus(design=array[1, 2, 1], grid=array[2, 1, 2], dimension_labels=['x', 'y'])"


def test_eq(stimulus: PRFStimulus):
    """Test equality of two PRFStimulus objects."""
    stimulus_2 = PRFStimulus(
        design=np.zeros((1, 2, 1)),
        grid=np.zeros((2, 1, 2)),
        dimension_labels=["x", "y"],
    )

    assert stimulus == stimulus_2


def test_ne(stimulus: PRFStimulus):
    """Test inequality of two PRFStimulus objects."""
    stimulus_3 = PRFStimulus(
        design=np.zeros((1, 2, 2)),
        grid=np.zeros((2, 2, 2)),
        dimension_labels=["x", "y"],
    )

    assert stimulus != stimulus_3


def test_ne_different_type(stimulus: PRFStimulus):
    """Test inequality of PRFStimulus object and object with different type."""
    assert stimulus != np.zeros((3, 3, 3))


def test_hash_error(stimulus: PRFStimulus):
    """Test that hashing raises an error."""
    with pytest.raises(TypeError):
        hash(stimulus)


@pytest.mark.parametrize(
    ("dimensions", "axis"),
    [("1D", "width"), ("2D", "width"), ("2D", "height"), ("3D", "depth")],
)
def test_rectangular_grid(dimensions: str, axis: str):
    """Check that rectangular grids can be created."""
    width = 10
    height = 20
    depth = 10

    scale = 2

    match axis:
        case "width":
            width *= scale
        case "height":
            height *= scale
        case "depth":
            depth *= scale

    num_frames = 10

    # 1D shape
    design_shape = (num_frames, width)
    grid_shape = (width, 1)

    match dimensions:
        case "2D":
            design_shape = (num_frames, width, height)
            grid_shape = (width, height, 2)
        case "3D":
            design_shape = (num_frames, width, height, depth)
            grid_shape = (width, height, depth, 3)

    design = np.zeros(design_shape)
    grid = np.zeros(grid_shape)

    stimulus = PRFStimulus(
        design=design,
        grid=grid,
    )

    assert isinstance(stimulus, PRFStimulus)


@pytest.mark.parametrize("direction", ["horizontal", "vertical"])
def test_create_2d_bar_stimulus(direction: str):
    """Check that a valid 2D bar stimulus can be created."""
    stimulus = PRFStimulus.create_2d_bar_stimulus(
        num_frames=10,
        width=5,
        height=4,
        direction=direction,
    )

    assert isinstance(stimulus, PRFStimulus)
    # First and last frame should be empty
    assert np.all(stimulus.design[0] == 0.0)
    assert np.all(stimulus.design[-1] == 0.0)

    # Design should have range 0-1
    assert np.min(stimulus.design) == 0.0
    assert np.max(stimulus.design) == 1.0

    match direction:
        case "horizontal":
            # For horizontal bars, each row (height axis) should have the same value within a frame
            # Each row in the frame should be constant
            rows_equal = np.all(stimulus.design == stimulus.design[:, [0], :])  # (10,5,4) == (10,1,4)
            assert np.all(rows_equal)
        case "vertical":
            # For vertical bars, each column (width axis) should have the same value within a frame
            # All values in each column should be equal for all frames
            cols_equal = np.all(stimulus.design == stimulus.design[:, :, [0]])
            assert np.all(cols_equal)


def test__get_grid_limits():
    """Test that grid limits get extracted correctly."""
    x = np.arange(-4, 4, 1)
    y = np.arange(-2, 6, 2)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack((xv, yv), axis=-1)
    result = _get_grid_limits(grid)
    expected = (-4.0, 3.0, -2.0, 4.0)
    assert result == expected, "Grid extent extracted incorrectly"


def test__verify_dimensions(stimulus_1d: PRFStimulus):
    """Test that error is raised."""
    with pytest.raises(StimulusDimensionError):
        _verify_dimensions(stimulus_1d, 2)


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
