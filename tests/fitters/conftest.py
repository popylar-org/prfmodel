"""Setup for fitter tests."""

import pandas as pd
import pytest
from prfmodel.examples import load_2d_bar_stimulus
from prfmodel.models.gaussian import Gaussian2DPRFModel

parametrize_dtype = pytest.mark.parametrize("dtype", [None, "float32"])


class TestSetup:
    """Setup parameters and objects for fitter tests."""

    @pytest.fixture
    def stimulus(self):
        """Stimulus object."""
        stimulus = load_2d_bar_stimulus()

        stimulus.design = stimulus.design[:25]

        return stimulus

    @pytest.fixture
    def model(self):
        """Gaussian 2D pRF model instance."""
        return Gaussian2DPRFModel()

    @pytest.fixture
    def params(self):
        """Parameters dataframe."""
        # 3 batches
        return pd.DataFrame(
            {
                "mu_x": [-1.0, 1.0, 0.0],
                "mu_y": [1.0, -1.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "shape": [6.0, 6.0, 6.0],
                "rate": [0.9, 0.9, 0.9],
                "shift": [5.0, 5.0, 5.0],
                "baseline": [0.0, 0.0, 0.0],
                "amplitude": [1.0, 1.0, 1.0],
            },
        )
