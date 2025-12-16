"""Test CFStimulus class."""

import numpy as np
import pytest

# Needs to be imported to recreate stimulus from repr
from numpy import array  # noqa: F401
from prfmodel.stimulus.cf import CFStimulus
from prfmodel.stimulus.cf import DistanceMatrixShapeError
from prfmodel.stimulus.cf import DistanceMatrixSourceShapeError


def test_distance_matrix_shape_error():
    """Test that DistanceMatrixShapeError is raised."""
    with pytest.raises(DistanceMatrixShapeError):
        _ = CFStimulus(
            distance_matrix=np.ones((3, 2)),
            source_response=np.zeros((3, 1)),
        )


def test_distance_matrix_source_shape_error():
    """Test that DistanceMatrixSourceShapeError is raised."""
    with pytest.raises(DistanceMatrixSourceShapeError):
        _ = CFStimulus(
            distance_matrix=np.ones((3, 3)),
            source_response=np.zeros((2, 1)),
        )


@pytest.fixture
def stimulus():
    """Stimulus object."""
    return CFStimulus(
        distance_matrix=np.ones((3, 3)),
        source_response=np.zeros((3, 1)),
    )


def test_repr(stimulus: CFStimulus):
    """Test machine-readable string representation of CFStimulus."""
    stimulus_2 = eval(repr(stimulus))  # noqa: S307

    assert stimulus == stimulus_2


def test_str(stimulus: CFStimulus):
    """Test human-readable string representation of CFStimulus."""
    assert str(stimulus) == "CFStimulus(distance_matrix=array[3, 3], source_response=array[3, 1])"


def test_eq(stimulus: CFStimulus):
    """Test equality of two CFStimulus objects."""
    stimulus_2 = CFStimulus(
        distance_matrix=np.ones((3, 3)),
        source_response=np.zeros((3, 1)),
    )

    assert stimulus == stimulus_2


def test_ne(stimulus: CFStimulus):
    """Test inequality of two CFStimulus objects."""
    stimulus_3 = CFStimulus(
        distance_matrix=np.ones((2, 2)),
        source_response=np.zeros((2, 1)),
    )

    assert stimulus != stimulus_3


def test_ne_different_type(stimulus: CFStimulus):
    """Test inequality of CFStimulus object and object with different type."""
    assert stimulus != np.zeros((3, 3, 3))


def test_hash_error(stimulus: CFStimulus):
    """Test that hashing raises error."""
    with pytest.raises(TypeError):
        hash(stimulus)
