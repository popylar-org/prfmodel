"""Tests for the shifted gamma distribution impulse response."""

from itertools import product
import numpy as np
import pandas as pd
import pytest
from prfmodel.models.impulse import ShiftedDerivativeGammaImpulse
from .conftest import TestImpulseSetup


class TestShiftedDerivativeGammaImpulse(TestImpulseSetup):
    """Tests for ShiftedDerivativeGammaImpulse class."""

    @pytest.fixture
    def parameter_range(self):
        """Range of parameters."""
        return np.round(np.linspace(0.2, 5.0, 3), 2)

    @pytest.fixture
    def shift_parameter_range(self):
        """Range of shift parameter."""
        return np.round(np.linspace(-5.0, 5.0, 5), 2)

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray, shift_parameter_range: np.ndarray):
        """Model parameter combinations."""
        values = np.array(
            list(
                product(
                    parameter_range,
                    parameter_range,
                    shift_parameter_range,
                ),
            ),
        )
        return pd.DataFrame.from_records(values, columns=["delay", "dispersion", "shift"])

    @pytest.fixture
    def irf_model(self):
        """Impulse response model object."""
        return ShiftedDerivativeGammaImpulse(self.duration, self.offset, self.resolution, self.norm)

    @pytest.fixture
    def irf_model_default(self):
        """Impulse response model object with default parameters."""
        default_params = {
            "delay": 6.0,
            "dispersion": 0.9,
            "shift": 5.0,
        }

        return ShiftedDerivativeGammaImpulse(self.duration, self.offset, self.resolution, self.norm, default_params)
