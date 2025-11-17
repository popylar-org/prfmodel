"""Tests for adapter and transformations."""

import warnings
import numpy as np
import pandas as pd
import pytest
from prfmodel.adapter import Adapter
from prfmodel.adapter import ParameterTransform
from prfmodel.fitters.backend.base import ParamsDict


@pytest.fixture
def params(num_rows: int = 10):
    """Parameters dictionary."""
    return {
        "x": np.linspace(-5, 5, num_rows),
        "y": np.linspace(0, 5, num_rows),
        "z": np.linspace(1, 5, num_rows),
    }


@pytest.fixture
def transform(request: pytest.FixtureRequest):
    """Transform object."""
    return ParameterTransform(
        request.param[0],
        request.param[1],
        request.param[2],
    )


@pytest.mark.parametrize("params_wrapper", [pd.DataFrame, ParamsDict])
@pytest.mark.parametrize("transform", [(["x"], np.log, np.exp), (["y", "z"], np.sqrt, np.square)], indirect=True)
def test_parameter_transform(transform: ParameterTransform, params_wrapper: type, params: pd.DataFrame):
    """Test that forward and backward transformation gives the correct result."""
    params = params_wrapper(params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result_forward = transform.forward(params)
        result_inverse = transform.inverse(params)

        for param in transform.parameter_names:
            ref_forward = transform.forward_fun(params[param])
            assert np.all(
                result_forward[param][np.isfinite(result_forward[param])] == ref_forward[np.isfinite(ref_forward)],
            ), "Forward transform does not give correct results"

            ref_inverse = transform.inverse_fun(params[param])
            assert np.all(
                result_inverse[param][np.isfinite(result_inverse[param])] == ref_inverse[np.isfinite(ref_inverse)],
            ), "Inverse transform does not give correct results"


@pytest.mark.parametrize("params_wrapper", [pd.DataFrame, ParamsDict])
def test_adapter(params_wrapper: type, params: pd.DataFrame):
    """Test that Adapter returns the correct object type."""
    adapter = Adapter(
        transforms=[
            ParameterTransform(["x"], np.log, np.exp),
            ParameterTransform(["y"], np.sqrt, np.log),
        ],
    )

    params = params_wrapper(params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        result_forward = adapter.forward(params)
        result_inverse = adapter.inverse(result_forward)

    assert isinstance(result_forward, params_wrapper)
    assert isinstance(result_inverse, params_wrapper)
