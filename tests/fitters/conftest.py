"""Setup for fitter tests."""

import sys
import keras
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
