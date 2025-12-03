"""Setup for fitter tests."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus

parametrize_dtype = pytest.mark.parametrize("dtype", [None, "float32"])


class TestSetup:
    """Setup parameters and objects for fitter tests."""

    width: int = 5
    height: int = 4
    num_frames: int = 10

    @pytest.fixture
    def stimulus(self):
        """Stimulus object."""
        y = np.linspace(-2, 2, num=self.height)
        x = np.linspace(-2, 2, num=self.width)
        xv, yv = np.meshgrid(x, y)
        grid = np.stack((xv, yv), axis=-1)

        design = np.ones((self.num_frames, self.height, self.width))

        return Stimulus(
            design=design,
            grid=grid,
            dimension_labels=["y", "x"],
        )

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
