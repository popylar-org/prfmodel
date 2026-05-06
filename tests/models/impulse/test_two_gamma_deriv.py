"""Tests for the weighted derivative difference of two gamma distribution impulse response."""

from itertools import product
import numpy as np
import pandas as pd
import pytest
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from .conftest import TestImpulseSetup


class TestDerivativeTwoGammaImpulse(TestImpulseSetup):
    """Tests for DerivativeTwoGammaImpulse class."""

    @pytest.fixture
    def parameter_range(self):
        """Range of parameters."""
        return np.round(np.linspace(0.2, 5.0, 3), 2)

    @pytest.fixture
    def ratio_range(self):
        """Range of weigth parameter."""
        return np.linspace(0.1, 0.9, 3)

    @pytest.fixture
    def weight_deriv_range(self):
        """Range of weigth parameter."""
        return np.linspace(-2.0, 2.0, 3)

    @pytest.fixture
    def parameters(self, parameter_range: np.ndarray, ratio_range: np.ndarray, weight_deriv_range: np.ndarray):
        """Model parameter combinations."""
        values = np.array(
            list(
                product(
                    parameter_range,
                    parameter_range,
                    parameter_range,
                    parameter_range,
                    ratio_range,
                    weight_deriv_range,
                ),
            ),
        )
        return pd.DataFrame.from_records(
            values,
            columns=["delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv"],
        )

    @pytest.fixture
    def irf_model(self):
        """Impulse response model object."""
        return DerivativeTwoGammaImpulse(self.duration, self.offset, self.resolution, self.norm)

    @pytest.fixture
    def irf_model_default(self):
        """Impulse response model object with default parameters."""
        default_params = {
            "delay": 6.0,
            "dispersion": 0.9,
            "undershoot": 12.0,
            "u_dispersion": 0.9,
            "ratio": 0.48,
            "weight_deriv": -0.5,
        }

        return DerivativeTwoGammaImpulse(self.duration, self.offset, self.resolution, self.norm, default_params)
