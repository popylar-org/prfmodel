"""Tests for adapter and transformations."""

import warnings
from collections.abc import Callable
import numpy as np
import pandas as pd
import pytest
from prfmodel.adapter import Adapter
from prfmodel.adapter import ParameterConstraint
from prfmodel.adapter import ParameterTransform
from prfmodel.utils import ParamsDict

parameterize_params_wrapper = pytest.mark.parametrize("params_wrapper", [pd.DataFrame, ParamsDict])


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


@parameterize_params_wrapper
@pytest.mark.parametrize("transform", [(["x"], np.log, np.exp), (["y", "z"], np.sqrt, np.square)], indirect=True)
def test_parameter_transform(transform: ParameterTransform, params_wrapper: type, params: dict):
    """Test that forward and backward transformation gives the correct result."""
    params = params_wrapper(params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result_forward = transform.forward(params)
        result_inverse = transform.inverse(params)

        for param in transform.parameter_names:
            ref_forward = np.asarray(transform.forward_fun(params[param]))
            result_forward_param = np.asarray(result_forward[param])
            np.testing.assert_allclose(
                result_forward_param,
                ref_forward,
                equal_nan=True,
                err_msg="Forward transform does not give correct results",
            )

            ref_inverse = np.asarray(transform.inverse_fun(params[param]))
            result_inverse_param = np.asarray(result_inverse[param])
            np.testing.assert_allclose(
                result_inverse_param[np.isfinite(result_inverse_param)],
                ref_inverse[np.isfinite(ref_inverse)],
                equal_nan=True,
                err_msg="Inverse transform does not give correct results",
            )


@parameterize_params_wrapper
def test_parameter_transform_forward_inverse(params_wrapper: type, params: dict):
    """Test that forward(inverse(input)) == input for valid transform ranges."""
    params = params_wrapper(params)
    transform = ParameterTransform(["z"], np.log, np.exp)
    result_forward = transform.forward(params)
    result_inverse = transform.inverse(result_forward)

    np.testing.assert_allclose(np.asarray(result_inverse["z"]), np.asarray(params["z"]))


@parameterize_params_wrapper
def test_parameter_constraint_lower(params_wrapper: type, params: dict):
    """Test that lower constraint gives correct result."""
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["z"],
        lower="x",
    )

    result_forward = transform.forward(params)

    np.testing.assert_array_less(np.asarray(result_forward["x"]), np.asarray(result_forward["z"]))


@parameterize_params_wrapper
def test_parameter_constraint_lower_forward_inverse(params_wrapper: type, params: dict):
    """Test that forward(inverse(input)) == input."""
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["z"],
        lower="x",
    )
    result_forward = transform.forward(params)
    result_inverse = transform.inverse(result_forward)

    np.testing.assert_allclose(np.asarray(result_inverse["z"]), np.asarray(params["z"]))


@parameterize_params_wrapper
def test_parameter_constraint_upper(params_wrapper: type, params: dict):
    """Test that upper constraint gives correct result."""
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["x"],
        upper="z",
    )

    result_forward = transform.forward(params)

    np.testing.assert_array_less(np.asarray(result_forward["x"]), np.asarray(result_forward["z"]))


@parameterize_params_wrapper
def test_parameter_constraint_upper_forward_inverse(params_wrapper: type, params: dict):
    """Test that forward(inverse(input)) == input."""
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["x"],
        lower="z",
    )
    result_forward = transform.forward(params)
    result_inverse = transform.inverse(result_forward)

    np.testing.assert_allclose(np.asarray(result_inverse["x"]), np.asarray(params["x"]), rtol=1e-6)


def test_parameter_constraint_lower_upper_error():
    """Test that providing lower and upper bound returns an error."""
    with pytest.raises(NotImplementedError):
        _ = ParameterConstraint(
            parameter_names=["x"],
            lower="y",
            upper="z",
        )


@parameterize_params_wrapper
@pytest.mark.parametrize("transform_fun", [lambda x: x**2, np.exp, np.log])
def test_parameter_constraint_upper_transform(params_wrapper: type, transform_fun: Callable, params: dict):
    """Test that upper constraint with transform function gives correct result."""
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["x"],
        upper="z",
        transform_fun=transform_fun,
    )

    result_forward = transform.forward(params)

    np.testing.assert_array_less(np.asarray(result_forward["x"]), transform_fun(np.asarray(result_forward["z"])))


@parameterize_params_wrapper
def test_parameter_constraint_lower_fixed(params_wrapper: type, params: dict):
    """Test that lower constraint with fixed value gives correct result."""
    fixed = -3.0
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["x"],
        lower=fixed,
    )

    result_forward = transform.forward(params)

    np.testing.assert_array_less(fixed, np.asarray(result_forward["x"]))


@parameterize_params_wrapper
def test_parameter_constraint_upper_fixed(params_wrapper: type, params: dict):
    """Test that upper constraint with fixed value gives correct result."""
    fixed = 3.0
    params = params_wrapper(params)
    transform = ParameterConstraint(
        parameter_names=["x"],
        upper=fixed,
    )

    result_forward = transform.forward(params)

    np.testing.assert_array_less(np.asarray(result_forward["x"]), fixed)


@parameterize_params_wrapper
def test_adapter(params_wrapper: type, params: dict):
    """Test that Adapter returns the correct object type."""
    adapter = Adapter(
        transforms=[
            ParameterTransform(["x"], np.log, np.exp),
            ParameterTransform(["y"], np.sqrt, np.log),
            ParameterConstraint(["x"], lower="z", transform_fun=np.abs),
        ],
    )

    params = params_wrapper(params)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        result_forward = adapter.forward(params)
        result_inverse = adapter.inverse(result_forward)

    assert isinstance(result_forward, params_wrapper)
    assert isinstance(result_inverse, params_wrapper)
