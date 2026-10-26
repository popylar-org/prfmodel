"""Validation shared by the fitters."""

import numpy as np
from prfmodel.exceptions import ShapeError
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM


def _validate_data_shape(
    data: Tensor | np.ndarray,
    num_units: int | None = None,
    num_frames: int | None = None,
) -> None:
    """Check that target data has shape `(num_units, num_frames)`.

    The number of dimensions is always checked; `num_units` and `num_frames` are only checked when given.

    """
    # 'numpy.shape' reads the 'shape' attribute when there is one, so a backend tensor is not copied to the host
    data_shape = tuple(np.shape(data))

    if len(data_shape) != _EXPECTED_NDIM:
        raise ShapeError("data", data_shape, "must have exactly 2 dimensions (num_units, num_frames)")  # noqa: EM101 (exception literal)

    if num_units is not None and data_shape[0] != num_units:
        raise ShapeError("data", data_shape, f"must have {num_units} rows, one for each row of the parameters")  # noqa: EM101 (exception literal)

    if num_frames is not None and data_shape[1] != num_frames:
        raise ShapeError(
            "data",  # noqa: EM101 (exception literal)
            data_shape,
            f"must have {num_frames} columns, one for each frame of the model prediction",
        )
