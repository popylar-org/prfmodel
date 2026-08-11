"""Setup for impulse model tests."""

from abc import ABC
from abc import abstractmethod
import numpy as np
import pandas as pd
import pytest
from prfmodel.impulse.base import BaseImpulse
from tests.models.conftest import parametrize_dtype


class TestImpulseSetup(ABC):
    """Parameters for impulse model tests."""

    duration = 32.0
    offset = 0.0001
    resolution = 1.0
    norm = "sum"

    @pytest.fixture
    @abstractmethod
    def irf_model(self):
        """Impulse model object."""

    @pytest.fixture
    @abstractmethod
    def irf_model_default(self):
        """Impulse model object with default parameters."""

    def test_num_frames(self, irf_model: BaseImpulse):
        """Test that property num_frames is correct."""
        assert irf_model.num_frames == int(self.duration / self.resolution)

    def test_frames(self, irf_model: BaseImpulse):
        """Test that get_frames returns correct shape."""
        assert irf_model.get_frames().shape == (1, irf_model.num_frames)

    @pytest.mark.parametrize("attribute", ["duration", "offset", "resolution"])
    def test_time_axis_is_read_only(self, irf_model: BaseImpulse, attribute: str):
        """Test that the attributes defining the time axis cannot be reassigned.

        `get_frames` caches its axis per dtype and never invalidates that cache, so an attribute that
        could be reassigned would leave the model returning frames for its previous configuration.

        """
        with pytest.raises(AttributeError):
            setattr(irf_model, attribute, 2.0)

    def test_norm_error(self, irf_model: BaseImpulse):
        """Test that an invalid norm argument raises an error."""
        with pytest.raises(ValueError):
            irf_model.__class__(norm="test")

    @parametrize_dtype
    def test_call(self, irf_model: BaseImpulse, parameters: pd.DataFrame, dtype: str):
        """Test that model response has correct shape."""
        resp = irf_model(parameters, dtype)

        assert resp.shape == (parameters.shape[0], irf_model.get_frames().shape[1])

    def test_call_default_parameters(
        self,
        irf_model: BaseImpulse,
        irf_model_default: BaseImpulse,
        parameters: pd.DataFrame,
    ):
        """Test that model with default parameters predicts correct response."""
        parameters_without_default = parameters.drop(columns=irf_model_default.default_parameters.keys())

        resp_without_default = irf_model_default(parameters_without_default)

        assert resp_without_default.shape == (parameters.shape[0], irf_model_default.get_frames().shape[1])

        parameters_with_default = parameters.copy()

        for key, val in irf_model_default.default_parameters.items():
            parameters_with_default[key] = val

        assert np.all(np.asarray(resp_without_default) == np.asarray(irf_model(parameters_with_default)))
