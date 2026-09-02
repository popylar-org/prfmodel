"""Container for connective field stimulus distance matrix and source signal."""

from dataclasses import dataclass
import numpy as np
from keras import ops
from prfmodel.exceptions import ShapeError
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .base import Stimulus
from .base import StimulusTensors

_DISTANCE_MATRIX_DIMENSIONS = 2


@dataclass(frozen=True, eq=False)
class CFStimulusTensors(StimulusTensors):
    """Tensor-holding counterpart of a :class:`~prfmodel.stimuli.CFStimulus`.

    Holds the connective field stimulus arrays as backend tensors. Should be created with
    :meth:`CFStimulus.to_tensors`.

    Parameters
    ----------
    distance_matrix : :data:`prfmodel.typing.Tensor`
        The :attr:`CFStimulus.distance_matrix` array as a tensor.
    source_response : :data:`prfmodel.typing.Tensor`
        The :attr:`CFStimulus.source_response` array as a tensor.

    """

    distance_matrix: Tensor
    source_response: Tensor


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
    ShapeError
        If the distance matrix is not a square 2D matrix.
    ShapeMismatchError
        If the source response has a first dimension with a different size than the number of rows or columns in the
        distance matrix.

    Examples
    --------
    >>> import numpy as np
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
            raise ShapeError("distance_matrix", self.distance_matrix.shape, "must be a square matrix")  # noqa: EM101 (exception literal)

    def _check_distance_matrix_source_shape(self) -> None:
        if not self.distance_matrix.shape[0] == self.source_response.shape[0]:
            raise ShapeMismatchError(
                "distance_matrix",  # noqa: EM101 (exception literal)
                self.distance_matrix.shape,
                "source_response",
                self.source_response.shape,
            )

    def to_tensors(self, dtype: str | None = None) -> CFStimulusTensors:
        """Convert the stimulus arrays into backend tensors.

        Parameters
        ----------
        dtype : str, optional
            The dtype to convert the stimulus arrays to. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        CFStimulusTensors
            The stimulus arrays as tensors.

        Examples
        --------
        >>> import numpy as np
        >>> stimulus = CFStimulus(
        ...     distance_matrix=np.zeros((10, 10)),
        ...     source_response=np.ones((10, 20)),
        ... )
        >>> tensors = stimulus.to_tensors("float32")
        >>> print(tuple(tensors.source_response.shape))
        (10, 20)

        """
        dtype = get_dtype(dtype)

        return CFStimulusTensors(
            distance_matrix=ops.convert_to_tensor(self.distance_matrix, dtype=dtype),
            source_response=ops.convert_to_tensor(self.source_response, dtype=dtype),
        )
