"""Setup for fitter tests."""

import pandas as pd
import pytest
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.models.impulse import ShiftedDerivativeGammaImpulse
from tests.conftest import StimulusSetup

parametrize_dtype = pytest.mark.parametrize("dtype", [None, "float32"])

parametrize_impulse_model = pytest.mark.parametrize("model", [None, {"delay": 6.0, "dispersion": 0.9}], indirect=True)


class TestSetup(StimulusSetup):
    """Setup parameters and objects for fitter tests."""

    @pytest.fixture
    def model(self, request: pytest.FixtureRequest):
        """Gaussian 2D pRF model instance."""
        # Only when fixture is parameterized we access the 'param' attribute
        default_parameters = request.param if hasattr(request, "param") else None

        return Gaussian2DPRFModel(
            impulse_model=ShiftedDerivativeGammaImpulse(
                default_parameters=default_parameters,
            ),
        )

    @pytest.fixture
    def params(self):
        """Parameters dataframe."""
        # 3 batches
        return pd.DataFrame(
            {
                "mu_x": [-1.0, 1.0, 0.0],
                "mu_y": [1.0, -1.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "delay": [6.0 / 0.9, 6.0 / 0.9, 6.0 / 0.9],
                "dispersion": [0.9, 0.9, 0.9],
                "shift": [5.0, 5.0, 5.0],
                "baseline": [0.0, 0.0, 0.0],
                "amplitude": [1.0, 1.0, 1.0],
            },
        )
