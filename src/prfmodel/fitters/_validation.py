"""Validation shared by the fitters."""

import warnings
from collections.abc import Sequence
import numpy as np
import pandas as pd
from keras import ops
from prfmodel.exceptions import InvalidUnitsWarning
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


def _find_valid_units(data: Tensor) -> np.ndarray:
    """Return the indices of the units in `data` that can be fit, and warn about the others.

    A unit cannot be fit when its time series is constant or contains NaN or infinite values.

    """
    # The tests run on the device and only the two boolean vectors are copied to the host
    is_nonfinite = ops.convert_to_numpy(ops.logical_not(ops.all(ops.isfinite(data), axis=1)))
    # A row of infinite values also compares equal to its first frame; it is only counted as non-finite
    is_constant = ops.convert_to_numpy(ops.all(ops.equal(data, data[:, :1]), axis=1)) & ~is_nonfinite

    num_constant = int(is_constant.sum())
    num_nonfinite = int(is_nonfinite.sum())

    if num_constant + num_nonfinite > 0:
        msg = (
            f"Excluded {num_constant + num_nonfinite} of {len(is_constant)} units from the fit: {num_constant} with "
            f"a constant time series and {num_nonfinite} with NaN or infinite values. Their estimated parameters "
            "and losses are NaN."
        )
        warnings.warn(msg, category=InvalidUnitsWarning, stacklevel=3)

    return np.flatnonzero(~(is_constant | is_nonfinite))


def _merge_unit_estimates(
    parameters: pd.DataFrame,
    estimates: pd.DataFrame,
    valid_units: np.ndarray,
    estimated_names: Sequence[str],
) -> pd.DataFrame:
    """Combine the estimates of the units that were fit with the parameters of the units that were excluded.

    `estimates` has the columns of `parameters` and one row for each unit in `valid_units`, in that order. For the
    excluded units, the columns in `estimated_names` are NaN and every other column keeps its value from
    `parameters`.

    """
    merged = parameters.copy()

    # Column by column and by position, because the estimates carry a fresh index that need not line up with the
    # one of 'parameters'. Each column takes the dtype of its estimates, so that excluded units do not change it.
    for name in estimates.columns:
        fitted = estimates[name].to_numpy()

        if name in estimated_names:
            column = np.full(len(parameters), np.nan, dtype=np.result_type(fitted.dtype, np.float32))
        else:
            column = parameters[name].to_numpy().astype(fitted.dtype)

        column[valid_units] = fitted
        merged[name] = column

    return merged
