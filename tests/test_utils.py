"""Tests for utility functions and classes."""

import keras
import numpy as np
import pytest
from prfmodel.utils import ParamsDict
from prfmodel.utils import UndefinedResponseWarning
from prfmodel.utils import _get_norm_fun
from prfmodel.utils import normalize_response


@pytest.mark.parametrize("norm", [None, "sum", "mean", "max", "norm"])
def test_normalize_response(norm: str):
    """Test that normalize_response returns correct result."""
    response = np.expand_dims(np.linspace(-5, 5, 100), 0)
    response_norm = np.asarray(normalize_response(response, norm=norm))

    assert response_norm.shape == response.shape

    if norm is not None:
        norm_fun = _get_norm_fun(norm)
        response_norm_ref = response / np.asarray(norm_fun(response, axis=1, keepdims=True))
    else:
        response_norm_ref = response

    assert np.allclose(response_norm, response_norm_ref)


def test_normalize_response_error():
    """Test that normalize_response raises an error for wrong input shape."""
    response = np.ones((10,))

    with pytest.raises(ValueError):
        normalize_response(response)

    response = 10

    with pytest.raises(ValueError):
        normalize_response(response)

    response = np.ones((10, 2, 1))

    with pytest.raises(ValueError):
        normalize_response(response)


def test_normalize_response_zero_norm():
    """Test that normalize response raises a warning for zero norms."""
    response = np.zeros((2, 10))

    with pytest.warns(UndefinedResponseWarning):
        response_norm = np.asarray(normalize_response(response))

    assert np.all(np.isnan(response_norm))


class TestParamsDict:
    """Tests for ParamsDict class."""

    shape: tuple[int] = (3, 1)

    @pytest.fixture
    def params_dict(self):
        """ParamsDict object."""
        return ParamsDict({"a": 0.0, "b": [1.0], "c": np.ones(self.shape[0]), "d": keras.ops.ones(self.shape)})

    def test_get_item(self, params_dict: ParamsDict):
        """Test that getting an item with a single key returns the correct shape and values."""
        for key in params_dict.columns:
            x = params_dict[key]
            assert x.shape == self.shape[:1]

        # torch requires us to convert tensors to numpy arrays before we can compare against floats
        assert np.all(np.asarray(params_dict["a"]) == 0.0)
        assert np.all(np.asarray(params_dict["b"]) == 1.0)

    def test_get_item_list(self, params_dict: ParamsDict):
        """Test that getting items with a list of keys returns the correct shapes and values."""
        x = params_dict[params_dict.columns]
        assert x.shape == (self.shape[0], len(params_dict.columns))

    def test_set_item(self, params_dict: ParamsDict):
        """Test that setting an item with a single key stores the correct shape and values."""
        new_params_dict = ParamsDict(
            {
                "e": keras.ops.zeros(self.shape),
            },
        )

        for key in params_dict.columns:
            new_params_dict[key] = params_dict[key]
            x = new_params_dict[key]
            assert x.shape == self.shape[:1]

    def test_set_item_list(self, params_dict: ParamsDict):
        """Test that setting items with a list of keys stores the correct shapes and values."""
        new_params_dict = ParamsDict(
            {
                "e": keras.ops.zeros(self.shape),
            },
        )
        new_params_dict[params_dict.columns] = params_dict[params_dict.columns]
        x = new_params_dict[params_dict.columns]
        assert x.shape == (self.shape[0], len(params_dict.columns))
