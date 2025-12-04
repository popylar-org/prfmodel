"""Global test configurations."""

import pandas as pd
import pytest
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from prfmodel.stimulus.prf import PRFStimulus

parametrize_impulse_model = pytest.mark.parametrize("model", [None, {"delay": 6.0, "dispersion": 0.9}], indirect=True)


class PRFStimulusSetup:
    """Test setup for pRF stimulus object."""

    start_frame: int = 40
    end_frame: int = 65

    @pytest.fixture
    def stimulus(self):
        """2D bar prF stimulus object."""
        stimulus = load_2d_prf_bar_stimulus()

        # Select subset of time frames that contain a single bar movement across screen
        design_sub = stimulus.design[self.start_frame : self.end_frame]

        # Stimulus is immutable so we need to recreated it
        return PRFStimulus(
            design=design_sub,
            grid=stimulus.grid,
            dimension_labels=stimulus.dimension_labels,
        )


class TestSetup(PRFStimulusSetup):
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
