"""Tests for grid fitting."""

import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters.grid import GridFitter
from prfmodel.fitters.grid import GridHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus
from .conftest import TestSetup
from .conftest import parametrize_dtype
from .conftest import parametrize_impulse_model


class TestGridFitter(TestSetup):
    """Tests for GridFitter class."""

    def _check_history(self, history: GridHistory) -> None:
        assert isinstance(history, GridHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)

    def _check_grid_params(self, result_params: pd.DataFrame, params: pd.DataFrame) -> None:
        assert isinstance(result_params, pd.DataFrame)
        assert result_params.shape == params.shape
        assert np.allclose(result_params, params)

    @pytest.fixture
    def param_ranges(self):
        """Parameter ranges."""
        return {
            "mu_x": list(range(-2, 3, 1)),
            "mu_y": list(range(-2, 3, 1)),
            "sigma": list(range(1, 4, 1)),
            "shape": [6.0],
            "rate": [0.9],
            "shift": [5.0],
            "baseline": [0.0],
            "amplitude": [1.0],
        }

    @parametrize_dtype
    @parametrize_impulse_model
    @pytest.mark.parametrize(
        "loss",
        [None, keras.losses.MeanSquaredError(reduction="none")],
    )
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        loss: keras.losses.Loss,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
        dtype: str,
    ):
        """Test that fit returns objects with the correct type and attributes."""
        fitter = GridFitter(
            model=model,
            stimulus=stimulus,
            loss=loss,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, grid_params = fitter.fit(observed, param_ranges, chunk_size=20)

        self._check_history(history)
        self._check_grid_params(grid_params, params)
