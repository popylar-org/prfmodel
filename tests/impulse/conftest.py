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
    norm: str | None = "sum"

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

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"duration": 0.0}, "'duration' must be > 0"),
            ({"duration": -1.0}, "'duration' must be > 0"),
            ({"resolution": 0.0}, "'resolution' must be > 0"),
            ({"resolution": -1.0}, "'resolution' must be > 0"),
        ],
    )
    def test_time_axis_arguments_are_checked_once(self, irf_model: BaseImpulse, kwargs: dict, match: str):
        """Test that an unusable time axis is rejected at construction rather than much later.

        Checking here costs one comparison per model instead of anything per evaluation, and it names the
        argument the user actually passed. It catches cases nothing caught before: `resolution=0.0` used
        to surface as a `ZeroDivisionError` from `num_frames`, and a negative `duration` produced an empty
        `(1, 0)` frame array.

        `offset` is deliberately absent: it is unconstrained, see `test_negative_offset_*` below.

        """
        with pytest.raises(ValueError, match=match):
            irf_model.__class__(**kwargs)

    def test_negative_offset_is_allowed(self, irf_model: BaseImpulse, parameters: pd.DataFrame):
        """Test that a negative offset is accepted and produces a finite response.

        The densities are zero below their support rather than NaN, so frames at or below zero are a
        legitimate way to express a pure lag -- the only way to, for a model with no `shift` parameter.

        """
        model = irf_model.__class__(offset=-5.0, resolution=1.0, norm=None)

        assert np.asarray(model.get_frames())[0, 0] == pytest.approx(-4.5)

        resp = np.asarray(model(parameters))

        assert np.all(np.isfinite(resp))

    def test_negative_offset_gives_the_kernel_leading_zeros(
        self,
        irf_model: BaseImpulse,
        parameters: pd.DataFrame,
    ):
        """Test that the frames at or below the model's support contribute exactly zero.

        `convolve_prf_impulse_response` treats index 0 of the kernel as lag 0, so these leading zeros are
        what turns a negative offset into a delay of `-offset / resolution` frames.

        The support starts at zero, except for a model with a `shift` parameter, which moves it -- a
        negative `shift` legitimately puts mass at a negative time.

        """
        model = irf_model.__class__(offset=-5.0, resolution=1.0, norm=None)

        frames = np.asarray(model.get_frames())
        support_start = (
            parameters[["shift"]].to_numpy() if "shift" in parameters.columns else np.zeros((len(parameters), 1))
        )
        outside_support = frames <= support_start

        resp = np.asarray(model(parameters))

        assert np.all(resp[outside_support] == 0.0)
        assert np.any(resp[~outside_support] != 0.0)
        assert outside_support[:, 0].any(), "the negative offset should put at least one frame outside"

    def test_sum_normalization_ignores_the_leading_zeros(
        self,
        irf_model: BaseImpulse,
        parameters: pd.DataFrame,
    ):
        """Test that a negative offset does not change what `norm="sum"` divides by beyond truncation."""
        model = irf_model.__class__(offset=-5.0, resolution=1.0, norm="sum")

        resp = np.asarray(model(parameters))

        # float32: the response is normalized in the model's dtype, not in float64
        np.testing.assert_allclose(resp.sum(axis=1), 1.0, rtol=1e-3)

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
