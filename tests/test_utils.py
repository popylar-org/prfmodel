"""Tests for utility functions and classes."""

import keras
import numpy as np
import pytest
from prfmodel.utils import ParamsDict


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
