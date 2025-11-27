"""Tests for the shifted gamma distribution impulse response."""

from itertools import product
import numpy as np
import pandas as pd
import pytest
from prfmodel.models.impulse import ShiftedGammaImpulse
from .conftest import TestImpulseSetup


class TestShiftedGammaImpulse(TestImpulseSetup):
    """Tests for ShiftedGammaImpulse class."""

    @pytest.fixture
    def parameter_range(self):
        """Range of parameters."""
        return np.round(np.linspace(0.1, 5.0, 3), 2)

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
        return pd.DataFrame.from_records(values, columns=["shape", "rate", "shift"])

    @pytest.fixture
    def irf_model(self):
        """Impulse response model object."""
        return ShiftedGammaImpulse(self.duration, self.offset, self.resolution)

    @pytest.fixture
    def irf_model_default(self):
        """Impulse response model object with default parameters."""
        default_params = {
            "shape": 6.0,
            "rate": 0.9,
            "shift": 5.0,
        }

        return ShiftedGammaImpulse(self.duration, self.offset, self.resolution, default_params)
