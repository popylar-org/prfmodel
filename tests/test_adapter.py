"""Tests for adapter and transformations."""

import numpy as np
import pandas as pd
import pytest
from keras import ops
from prfmodel.fitters.adapter import Adapter
from prfmodel.fitters.adapter import ParameterConstraint
from prfmodel.fitters.adapter import ParameterTransform
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
@pytest.mark.parametrize("transform", [(["x"], ops.log, ops.exp), (["y", "z"], ops.sqrt, ops.square)], indirect=True)
def test_parameter_transform(transform: ParameterTransform, params_wrapper: type, params: dict):
    """Test that transformation and inverse give the correct results."""
    params = params_wrapper(params)

    result_transformed = transform.transform(params)
    result_inverse = transform.inverse(params)

    for param in transform.parameter_names:
        ref_transformed = np.asarray(transform.transform_fun(params[param]))
        result_transformed_param = np.asarray(result_transformed[param])
        np.testing.assert_allclose(
            result_transformed_param,
            ref_transformed,
            equal_nan=True,
            err_msg="Transform does not give correct results",
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
def test_parameter_transform_inverse(params_wrapper: type, params: dict):
    """Test that transform(inverse(input)) == input for valid transform ranges."""
    params = params_wrapper(params)
    transform = ParameterTransform(["z"], ops.log, ops.exp)
    result_transformed = transform.transform(params)
    result_inverse = transform.inverse(result_transformed)

    np.testing.assert_allclose(np.asarray(result_inverse["z"]), np.asarray(params["z"]))


@pytest.fixture
def bounded_params(num_rows: int = 10):
    """Parameters lying strictly inside the ``low``/``high`` bounds used by the constraint tests."""
    return {
        "x": np.linspace(2.0, 6.0, num_rows),
        "low": np.linspace(0.0, 1.0, num_rows),
        "high": np.linspace(7.0, 9.0, num_rows),
    }


@pytest.fixture
def unbounded_params(num_rows: int = 10):
    """Values on the unbounded (optimization) scale, over a range where float32 does not saturate.

    Far enough from zero the exponential underflows relative to the bound and the natural-scale value
    rounds onto the bound exactly (at negative values for a lower bound, positive for an upper one).
    For bounds of this magnitude that happens beyond roughly +/-14, so the range stays inside it; the
    saturating regime is covered by `unbounded_params_extreme`.

    """
    return {
        "x": np.linspace(-12.0, 12.0, num_rows),
        "low": np.linspace(0.0, 1.0, num_rows),
        "high": np.linspace(7.0, 9.0, num_rows),
    }


@pytest.fixture
def unbounded_params_extreme(num_rows: int = 10):
    """Values on the unbounded scale far enough out that float32 saturates onto the bound."""
    return {
        "x": np.linspace(-60.0, 60.0, num_rows),
        "low": np.linspace(0.0, 1.0, num_rows),
        "high": np.linspace(7.0, 9.0, num_rows),
    }


# `transform` maps the natural (bounded) scale onto the unbounded scale the optimizer works on, and
# `inverse` maps back. The bound is therefore enforced by `inverse`, not by `transform`.


@parameterize_params_wrapper
@pytest.mark.parametrize(("bound_kwargs", "bound_key"), [({"lower": "low"}, "low"), ({"lower": 1.5}, None)])
def test_parameter_constraint_lower_inverse_enforces_bound(
    params_wrapper: type,
    bound_kwargs: dict,
    bound_key: str | None,
    unbounded_params: dict,
):
    """Test that mapping any unbounded value back yields a value above the lower bound."""
    params = params_wrapper(unbounded_params)
    constraint = ParameterConstraint(parameter_names=["x"], **bound_kwargs)

    result = constraint.inverse(params)

    bound = np.asarray(result[bound_key]) if bound_key else bound_kwargs["lower"]
    np.testing.assert_array_less(bound, np.asarray(result["x"]))


@parameterize_params_wrapper
@pytest.mark.parametrize(("bound_kwargs", "bound_key"), [({"upper": "high"}, "high"), ({"upper": 8.5}, None)])
def test_parameter_constraint_upper_inverse_enforces_bound(
    params_wrapper: type,
    bound_kwargs: dict,
    bound_key: str | None,
    unbounded_params: dict,
):
    """Test that mapping any unbounded value back yields a value below the upper bound."""
    params = params_wrapper(unbounded_params)
    constraint = ParameterConstraint(parameter_names=["x"], **bound_kwargs)

    result = constraint.inverse(params)

    bound = np.asarray(result[bound_key]) if bound_key else bound_kwargs["upper"]
    np.testing.assert_array_less(np.asarray(result["x"]), bound)


@parameterize_params_wrapper
@pytest.mark.parametrize(
    ("bound_kwargs", "bound_key", "side"),
    [
        ({"lower": "low"}, "low", "lower"),
        ({"lower": 1.5}, None, "lower"),
        ({"upper": "high"}, "high", "upper"),
        ({"upper": 8.5}, None, "upper"),
    ],
)
def test_parameter_constraint_inverse_never_violates_bound(
    params_wrapper: type,
    bound_kwargs: dict,
    bound_key: str | None,
    side: str,
    unbounded_params_extreme: dict,
):
    """Test that even extreme unbounded values never map to the wrong side of the bound.

    This is the safety property an optimizer depends on: whatever it proposes, the model never sees a
    value that violates the constraint, and never a NaN. Far from the bound the exponential underflows
    in float32 and the result rounds onto the bound exactly, so the comparison is not strict here.

    """
    params = params_wrapper(unbounded_params_extreme)
    constraint = ParameterConstraint(parameter_names=["x"], **bound_kwargs)

    result = np.asarray(constraint.inverse(params)["x"])

    assert np.all(np.isfinite(result)), "Inverse must never produce NaN or infinity"

    bound = np.asarray(params_wrapper(dict(unbounded_params_extreme))[bound_key]) if bound_key else bound_kwargs[side]

    if side == "lower":
        assert np.all(result >= bound), "Inverse produced a value below the lower bound"
    else:
        assert np.all(result <= bound), "Inverse produced a value above the upper bound"


@parameterize_params_wrapper
@pytest.mark.parametrize(
    "bound_kwargs",
    [
        {"lower": "low"},
        {"lower": 1.5},
        {"upper": "high"},
        {"upper": 8.5},
        {"lower": "low", "bound_fun": ops.square},
        {"upper": "high", "bound_fun": ops.square},
    ],
)
def test_parameter_constraint_round_trip(params_wrapper: type, bound_kwargs: dict, bounded_params: dict):
    """Test that inverse(transform(input)) == input for values strictly inside the bound."""
    params = params_wrapper(bounded_params)
    constraint = ParameterConstraint(parameter_names=["x"], **bound_kwargs)

    result = constraint.inverse(constraint.transform(params))

    np.testing.assert_allclose(
        np.asarray(result["x"]),
        np.asarray(bounded_params["x"]),
        # Loose enough for float32: a bound much larger than the value costs precision to cancellation
        rtol=1e-5,
        err_msg="Constraint does not round-trip",
    )


@parameterize_params_wrapper
@pytest.mark.parametrize(
    "bound_kwargs",
    [
        pytest.param({"lower": 2.0}, id="on-lower-bound"),
        pytest.param({"lower": 3.0}, id="below-lower-bound"),
        pytest.param({"upper": 6.0}, id="on-upper-bound"),
        pytest.param({"upper": 5.0}, id="above-upper-bound"),
    ],
)
def test_parameter_constraint_transform_rejects_values_outside_bound(
    params_wrapper: type,
    bound_kwargs: dict,
    bounded_params: dict,
):
    """Test that transforming a value on or beyond an open bound raises instead of returning an infinity."""
    params = params_wrapper(bounded_params)
    constraint = ParameterConstraint(parameter_names=["x"], **bound_kwargs)

    with pytest.raises(ValueError, match="strictly"):
        _ = constraint.transform(params)


@parameterize_params_wrapper
def test_parameter_constraint_bound_fun_applied(params_wrapper: type, bounded_params: dict):
    """Test that `bound_fun` is applied to the bound before it is used."""
    # low is in [0, 1], so low + 1 is in [1, 2] and x (in [2, 6]) is still strictly above it
    params = params_wrapper(dict(bounded_params))
    constraint = ParameterConstraint(parameter_names=["x"], lower="low", bound_fun=lambda b: b + 1.0)

    result = constraint.inverse(params)

    np.testing.assert_array_less(np.asarray(result["low"]) + 1.0, np.asarray(result["x"]))


def test_parameter_constraint_no_bound_error():
    """Test that providing neither bound returns an error."""
    with pytest.raises(ValueError, match="Either a lower or an upper bound"):
        _ = ParameterConstraint(parameter_names=["x"])


def test_parameter_constraint_lower_upper_error():
    """Test that providing lower and upper bound returns an error."""
    with pytest.raises(NotImplementedError):
        _ = ParameterConstraint(
            parameter_names=["x"],
            lower="y",
            upper="z",
        )


@parameterize_params_wrapper
def test_parameter_constraint_value_error(params_wrapper: type, params: dict):
    """Test that missing string bound names return an error."""
    params = params_wrapper(params)

    transform_lower = ParameterConstraint(
        parameter_names=["x"],
        lower="foo",
    )
    transform_upper = ParameterConstraint(
        parameter_names=["x"],
        upper="bar",
    )

    with pytest.raises(ValueError, match="dynamic"):
        _ = transform_lower.transform(params)

    with pytest.raises(ValueError, match="dynamic"):
        _ = transform_lower.inverse(params)

    with pytest.raises(ValueError, match="dynamic"):
        _ = transform_upper.transform(params)

    with pytest.raises(ValueError, match="dynamic"):
        _ = transform_upper.inverse(params)


@parameterize_params_wrapper
def test_adapter(params_wrapper: type, bounded_params: dict):
    """Test that Adapter returns the correct object type and round-trips through mixed transforms."""
    adapter = Adapter(
        transforms=[
            ParameterTransform(["x"], ops.log, ops.exp),
            ParameterTransform(["high"], ops.sqrt, ops.square),
            # Applied after the log above, so this constrains log(x), which stays above 'low'
            ParameterConstraint(["x"], lower="low"),
        ],
    )

    params = params_wrapper(dict(bounded_params))

    result_transformed = adapter.transform(params)
    result_inverse = adapter.inverse(result_transformed)

    assert isinstance(result_transformed, params_wrapper)
    assert isinstance(result_inverse, params_wrapper)

    for name in ["x", "high"]:
        np.testing.assert_allclose(
            np.asarray(result_inverse[name]),
            np.asarray(bounded_params[name]),
            rtol=1e-6,
            err_msg=f"Adapter does not round-trip '{name}'",
        )
