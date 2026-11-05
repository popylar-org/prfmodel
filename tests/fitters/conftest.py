"""Setup for fitter tests."""

import sys
import keras
import numpy as np
import pandas as pd
import pytest

# For the torch backend, the regression test fails due to slight numeric differences that cannot easily be captured
# with tolerances
skip_torch = pytest.mark.skipif(
    keras.backend.backend() == "torch",
    reason="Slight numerical differences in parameter estimates with torch backend",
)

skip_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Slight numerical differences in parameter estimates on Windows",
)

parametrize_dtype = pytest.mark.parametrize("dtype", [None, "float32", "float64"])

# The message of the 'InvalidUnitsWarning' for the data from 'insert_invalid_units'
INVALID_UNITS_MESSAGE = "Excluded 5 of 8 units from the fit: 2 with a constant time series and 3 with NaN or infinite"


def insert_invalid_units(observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleave units that cannot be fit with the three units in `observed`.

    The five inserted units are two constant ones (all zeros and all fives) and three non-finite ones (all NaN, a
    single NaN and all infinite). Returns the data and a mask that is true for the rows that come from `observed`.

    """
    single_nan = np.array(observed[0], dtype=np.float64)
    single_nan[len(single_nan) // 2] = np.nan

    num_frames = observed.shape[1]
    invalid = [
        np.zeros(num_frames),
        np.full(num_frames, 5.0),
        np.full(num_frames, np.nan),
        single_nan,
        np.full(num_frames, np.inf),
    ]

    rows = []
    is_valid = []
    for index in range(max(len(observed), len(invalid))):
        if index < len(observed):
            rows.append(observed[index])
            is_valid.append(True)
        if index < len(invalid):
            rows.append(invalid[index])
            is_valid.append(False)

    return np.stack(rows), np.array(is_valid)


def insert_invalid_parameters(params: pd.DataFrame, is_valid: np.ndarray, nan_names: list[str]) -> pd.DataFrame:
    """Expand `params` to the rows of the data from `insert_invalid_units`.

    The excluded units copy the parameters of the unit before them, with the parameters in `nan_names` set to NaN,
    as a grid search returns them for those units.

    """
    expanded = params.iloc[np.maximum(np.cumsum(is_valid) - 1, 0)].reset_index(drop=True)
    expanded.loc[~is_valid, nan_names] = np.nan

    return expanded
