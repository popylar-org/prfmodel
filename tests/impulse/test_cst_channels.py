"""Tests for the compressive spatiotemporal (CST) temporal channel impulse responses."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.impulse import SustainedImpulse
from prfmodel.impulse import TransientImpulse
from .conftest import TestImpulseSetup


class TestSustainedImpulsePeak:
    """The `time_to_peak` parameter must be the time at which the response peaks."""

    @pytest.mark.parametrize("time_to_peak", [0.05, 0.1, 0.23])
    def test_response_peaks_at_time_to_peak(self, time_to_peak: float):
        """Test that the sustained response peaks at `time_to_peak` seconds."""
        resolution = 0.001
        irf_model = SustainedImpulse(duration=2.0, resolution=resolution, norm=None)

        resp = np.asarray(irf_model(pd.DataFrame({"time_to_peak": [time_to_peak]})))
        peak_time = np.asarray(irf_model.get_frames())[0, resp.argmax()]

        assert peak_time == pytest.approx(time_to_peak, abs=resolution)


class TestTransientImpulseShape:
    """The transient channel is the biphasic difference of an excitatory and a slower inhibitory gamma."""

    duration = 5.0
    resolution = 0.001
    time_to_peak = 0.05

    @pytest.fixture
    def parameters(self):
        """Parameters for a single unit."""
        return pd.DataFrame({"time_to_peak": [self.time_to_peak]})

    @pytest.fixture
    def transient(self, parameters: pd.DataFrame):
        """Transient impulse response of a single unit."""
        irf_model = TransientImpulse(duration=self.duration, resolution=self.resolution, norm=None)

        return np.asarray(irf_model(parameters))[0]

    def test_response_is_biphasic(self, transient: np.ndarray):
        """Test that the response has both a positive and a negative lobe."""
        assert transient.max() > 0
        assert transient.min() < 0

    def test_response_integrates_to_zero(self, transient: np.ndarray):
        """Test that the two gamma densities cancel, since each integrates to one."""
        assert transient.sum() * self.resolution == pytest.approx(0.0, abs=1e-3)

    def test_response_crosses_zero_once(self, transient: np.ndarray):
        """Test that the response is truly biphasic rather than oscillating."""
        num_crossings = np.count_nonzero(np.diff(np.signbit(transient[transient != 0])))

        assert num_crossings == 1

    def test_peaks_earlier_than_sustained(self, parameters: pd.DataFrame):
        """Test that the sustained peak is later than the transient peak.

        The inhibitory component peaks after the excitatory one, so subtracting it pulls the transient peak
        forward in time relative to the sustained channel.

        """
        transient_model = TransientImpulse(duration=self.duration, resolution=self.resolution, norm=None)
        sustained_model = SustainedImpulse(duration=self.duration, resolution=self.resolution, norm=None)

        transient = np.asarray(transient_model(parameters))[0]
        sustained = np.asarray(sustained_model(parameters))[0]

        assert transient.argmax() < sustained.argmax()

    def test_inhibitory_component_peaks_later_than_excitatory(self, parameters: pd.DataFrame):
        """Test that the inhibitory lobe peaks at the ratio the reference prescribes."""
        irf_model = TransientImpulse(duration=self.duration, resolution=self.resolution, norm=None)
        frames = np.asarray(irf_model.get_frames())[0]
        transient = np.asarray(irf_model(parameters))[0]

        # The trough of the biphasic response is dominated by the inhibitory component
        expected_ratio = (irf_model.inhibitory_shape - 1) * irf_model.inhibitory_time_constant_ratio
        expected_ratio /= irf_model.shape - 1

        assert frames[transient.argmin()] > self.time_to_peak * expected_ratio


class TestTransientImpulseNormalization:
    """`norm` must be left at None for the transient channel."""

    def test_sum_normalization_silently_amplifies_the_kernel(self):
        """Test that `norm="sum"` blows the transient kernel up instead of raising or warning.

        The two gamma components each integrate to one, so their difference sums to a value near zero that is
        not exactly zero. `normalize_response` only warns when the norm is exactly zero, so dividing by that
        residual passes silently while scaling the kernel by orders of magnitude. This is the reason `norm`
        defaults to None here, and the reason this behaviour is pinned rather than left to be rediscovered.

        """
        parameters = pd.DataFrame({"time_to_peak": [5.0]})

        unnormalized = np.asarray(TransientImpulse(norm=None)(parameters))
        normalized = np.asarray(TransientImpulse(norm="sum")(parameters))

        assert unnormalized.sum() != 0.0
        assert np.abs(normalized).max() > 100 * np.abs(unnormalized).max()

    def test_default_norm_is_none(self):
        """Test that the transient channel is unnormalized by default."""
        assert TransientImpulse().norm is None
        assert SustainedImpulse().norm is None


class TestChannelsAreUnitWise:
    """Each unit's response must be built from that unit's own `time_to_peak`."""

    @pytest.mark.parametrize("impulse_class", [SustainedImpulse, TransientImpulse])
    def test_units_are_predicted_independently(self, impulse_class: type):
        """Test that a row of `time_to_peak` values produces one differently-timed response per row.

        `gamma_density` requires its shape and scale arguments to have matching shapes, so the scalar shape
        constant is broadcast against the `(num_units, 1)` parameter column. A broadcast that collapsed the unit
        axis would make these rows identical.

        """
        time_to_peak = [4.0, 6.0, 8.0]
        irf_model = impulse_class(duration=32.0, resolution=0.5, norm=None)

        resp = np.asarray(irf_model(pd.DataFrame({"time_to_peak": time_to_peak})))

        assert resp.shape[0] == len(time_to_peak)
        assert len(np.unique(resp.argmax(axis=1))) == len(time_to_peak)
        # Later peak times must produce later peaks
        assert np.all(np.diff(resp.argmax(axis=1)) > 0)


class TestSustainedImpulse(TestImpulseSetup):
    """Shared impulse model tests for SustainedImpulse."""

    norm = "sum"

    @pytest.fixture
    def parameters(self):
        """Model parameter combinations.

        `time_to_peak` is scaled to the one-second `resolution` of the shared suite rather than to the
        millisecond values the reference reports, so that the peak falls on the sampled time axis.

        """
        return pd.DataFrame({"time_to_peak": np.round(np.linspace(4.0, 10.0, 4), 2)})

    @pytest.fixture
    def irf_model(self):
        """Impulse model object."""
        return SustainedImpulse(self.duration, self.offset, self.resolution, self.norm)

    @pytest.fixture
    def irf_model_default(self):
        """Impulse model object with default parameters."""
        return SustainedImpulse(self.duration, self.offset, self.resolution, self.norm, {"time_to_peak": 6.0})


class TestTransientImpulse(TestImpulseSetup):
    """Shared impulse model tests for TransientImpulse."""

    norm = None  # Skips tests for sum normalization

    @pytest.fixture
    def parameters(self):
        """Model parameter combinations. See `TestSustainedImpulse.parameters` for the scaling."""
        return pd.DataFrame({"time_to_peak": np.round(np.linspace(4.0, 10.0, 4), 2)})

    @pytest.fixture
    def irf_model(self):
        """Impulse model object."""
        return TransientImpulse(self.duration, self.offset, self.resolution, self.norm)

    @pytest.fixture
    def irf_model_default(self):
        """Impulse model object with default parameters."""
        return TransientImpulse(self.duration, self.offset, self.resolution, self.norm, {"time_to_peak": 6.0})
