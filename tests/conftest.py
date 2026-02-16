"""Global test configurations."""

import pandas as pd
import pytest
from prfmodel.examples import load_2d_bar_stimulus
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.models.impulse import DerivativeTwoGammaImpulse

parametrize_impulse_model = pytest.mark.parametrize("model", [None, {"delay": 6.0, "dispersion": 0.9}], indirect=True)


class StimulusSetup:
    """Test setup for stimulus object."""

    start_frame: int = 40
    end_frame: int = 65

    @pytest.fixture
    def stimulus(self):
        """2D bar stimulus object."""
        stimululus = load_2d_bar_stimulus()

        # Select subset of time frames that contain a single bar movement across screen
        stimululus.design = stimululus.design[self.start_frame : self.end_frame]

        return stimululus


class TestSetup(StimulusSetup):
    """Setup parameters and objects for fitter tests."""

    @pytest.fixture
    def model(self, request: pytest.FixtureRequest):
        """Gaussian 2D pRF model instance."""
        # Only when fixture is parameterized we access the 'param' attribute
        default_parameters = request.param if hasattr(request, "param") else None

        return Gaussian2DPRFModel(
            impulse_model=DerivativeTwoGammaImpulse(
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
                "delay": [6.0, 6.0, 6.0],
                "dispersion": [0.9, 0.9, 0.9],
                "undershoot": [12.0, 12.0, 12.0],
                "u_dispersion": [0.9, 0.9, 0.9],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "baseline": [0.1, -0.1, 0.5],
                "amplitude": [-2.0, 1.2, 0.1],
            },
        )
