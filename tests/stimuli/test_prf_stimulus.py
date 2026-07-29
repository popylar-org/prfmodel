"""Test PRFStimulus class."""

import numpy as np
import pytest

# Needs to be imported to recreate stimulus from repr
from numpy import array  # noqa: F401
from prfmodel.exceptions import ShapeError
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.stimuli import PRFStimulus


def test_grid_design_shape_error():
    """Test that shape mismatches are detected correctly by PRFStimulus."""
    with pytest.raises(ShapeMismatchError):
        _ = PRFStimulus(
            design=np.zeros((1, 2)),
            grid=np.zeros((1, 1)),
        )


def test_grid_dimension_error():
    """Test that grid axis-count vs last-dim mismatches are detected correctly by PRFStimulus."""
    with pytest.raises(ShapeError):
        _ = PRFStimulus(
            design=np.zeros((1, 1, 2)),
            grid=np.zeros((1, 2, 1)),
        )


def test_dimension_labels_error():
    """Test that label-count vs grid-dim mismatches are detected correctly by PRFStimulus."""
    with pytest.raises(ValueError, match="dimension_labels"):
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
