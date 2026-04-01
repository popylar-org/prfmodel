"""Container for connective field stimulus distance matrix and source signal."""

from dataclasses import dataclass
import numpy as np
from .base import Stimulus

_DISTANCE_MATRIX_DIMENSIONS = 2


class DistanceMatrixShapeError(Exception):
    """
    Exception raise when distance matrix is not a matrix.

    Parameters
    ----------
    distance_matrix_shape : tuple of int
        Shape of the distance matrix.

    """

    def __init__(self, distance_matrix_shape: tuple[int, ...]):
        super().__init__(
            f"'distance_matrix' must be a matrix but has shape {distance_matrix_shape}",
        )


class DistanceMatrixSourceShapeError(Exception):
    """
    Exception raised when the shapes of the distance matrix and source response do not match.

    Parameters
    ----------
    distance_matrix_shape : tuple of int
        Shape of the distance matrix.
    source_response_shape : tuple of int
        Shape of the source response array.

    """

    def __init__(self, distance_matrix_shape: tuple[int, ...], source_response_shape: tuple[int, ...]):
        super().__init__(
            f"'Shapes of 'distance_matrix' {distance_matrix_shape} and 'source_response' {source_response_shape} do not match",  # noqa: E501 (line too long)
        )


@dataclass(frozen=True, eq=False)
class CFStimulus(Stimulus):
    """
    Container for a connective field stimulus distance matrix and source response.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        A matrix with distances between source units.
    source_response : numpy.ndarray
        Array with responses for each source unit with shape `(num_units, num_frames)`. `num_units` is the number of
        source units and `num_frames` the number of time frames for each source response. `num_units` must match the
        number of rows and columns in `distance_matrix`.

    Raises
    ------
    DistanceMatrixShapeError
        If the distance matrix has more or less than two dimensions or dimensions of different size.
    DistanceMatrixSourceShapeError
        If the source response has a first dimension with a different size than the number of rows or columns in the
        distance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.stimuli.cf import CFStimulus
    >>> num_vertices, num_frames = 10, 20
    >>> distance_matrix = np.zeros((num_vertices, num_vertices))
    >>> source_response = np.ones((num_vertices, num_frames))
    >>> stimulus = CFStimulus(distance_matrix=distance_matrix, source_response=source_response)
    >>> print(stimulus.distance_matrix.shape)
    (10, 10)
    >>> print(stimulus.source_response.shape)
    (10, 20)

    """

    distance_matrix: np.ndarray
    source_response: np.ndarray

    def __post_init__(self):
        self._check_distance_matrix_shape()
        self._check_distance_matrix_source_shape()

    def _check_distance_matrix_shape(self) -> None:
        if (
            len(self.distance_matrix.shape) != _DISTANCE_MATRIX_DIMENSIONS
            or self.distance_matrix.shape[0] != self.distance_matrix.shape[1]
        ):
            raise DistanceMatrixShapeError(self.distance_matrix.shape)

    def _check_distance_matrix_source_shape(self) -> None:
        if not self.distance_matrix.shape[0] == self.source_response.shape[0]:
            raise DistanceMatrixSourceShapeError(self.distance_matrix.shape, self.source_response.shape)
