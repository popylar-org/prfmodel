"""Tests for grid fitting."""

import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters import GridFitter
from prfmodel.fitters import GridHistory
from prfmodel.fitters._grid import InfiniteLossWarning
from prfmodel.fitters.losses import CorrelationLoss
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from tests.conftest import TestSetup
from tests.conftest import parametrize_impulse_model
from tests.reference import _oracle
from .conftest import parametrize_dtype


class TestGridFitter(TestSetup):
    """Tests for GridFitter class."""

    def _check_history(self, history: GridHistory) -> None:
        assert isinstance(history, GridHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)

    def _check_grid_params(self, result_params: pd.DataFrame, params: pd.DataFrame, check_params: list[str]) -> None:
        assert isinstance(result_params, pd.DataFrame)
        assert result_params.shape == params.shape
        assert np.allclose(result_params[check_params], params[check_params], equal_nan=True)

    @pytest.fixture
    def param_ranges(self):
        """Parameter ranges.

        The data-generating parameters need to be included in the grid for the grid search to exactly recover them.

        """
        return {
            "mu_x": list(range(-2, 3, 1)),
            "mu_y": list(range(-2, 3, 1)),
            "sigma": list(range(1, 4, 1)),
            "delay": [6.0],
            "dispersion": [0.9],
            "undershoot": [12.0],
            "u_dispersion": [0.9],
            "ratio": [0.48],
            "weight_deriv": [0.5],
            "baseline": [0.1, -0.1, 0.5],
            "amplitude": [-2.0, 1.2, 0.1],
        }

    @parametrize_dtype
    @parametrize_impulse_model
    @pytest.mark.parametrize(
        ("loss", "check_params"),
        [
            # Correlation loss (default) ignores differences in baseline and amplitude
            (None, ["mu_x", "mu_y", "sigma"]),
            (CorrelationLoss(reduction="none"), ["mu_x", "mu_y", "sigma"]),
            # MSE is sensitive to baseline and amplitude
            (keras.losses.MeanSquaredError(reduction="none"), ["mu_x", "mu_y", "sigma", "baseline", "amplitude"]),
        ],
    )
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        loss: keras.losses.Loss,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
        check_params: list[str],
        dtype: str,
    ):
        """Test that the grid search recovers the parameters an independent model generated data from.

        The target comes from `tests.reference._oracle`, a plain NumPy/SciPy implementation of the
        forward pipeline, rather than from `model` itself. Generating it with `model` would make the
        recovery circular: any error in the forward model would appear in both the data and the
        search, cancel exactly, and leave the true parameters at the optimum regardless.

        A grid search is the most forgiving place to do this, because its output is discrete -- small
        numerical differences between the two implementations cannot move the result off the correct
        grid point.

        """
        fitter = GridFitter(
            model=model,
            stimulus=stimulus,
            loss=loss,
            dtype=dtype,
        )

        observed = _oracle.predict(stimulus, params, frames=_oracle.impulse_frames())

        history, grid_params = fitter.fit(observed, param_ranges, batch_size=20)

        self._check_history(history)
        self._check_grid_params(grid_params, params, check_params)

    def test_fit_default_loss_is_offset_invariant(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
    ):
        """Test that the default loss recovers the pRF from data with a baseline that is absent from the grid."""
        fitter = GridFitter(
            model=model,
            stimulus=stimulus,
        )

        shifted_params = params.copy()
        shifted_params["baseline"] += 10.0
        observed = model(stimulus, shifted_params)

        # The data baseline is far outside the grid, which only offers baselines close to zero
        offset_ranges = {**param_ranges, "baseline": [0.0]}

        history, grid_params = fitter.fit(observed, offset_ranges, batch_size=20)

        self._check_history(history)
        self._check_grid_params(grid_params, params, ["mu_x", "mu_y", "sigma"])

    def test_fit_infinite_loss_warning(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
    ):
        """Test that fit returns an infinite loss warning and matching NaN estimates when appropriate."""
        fitter = GridFitter(
            model=model,
            stimulus=stimulus,
        )

        params_copy = params.copy()
        params_copy.iloc[0, :] = np.nan

        observed = np.array(model(stimulus, params))  # Need to convert to numpy to assign value
        observed[0, :] = np.nan

        with pytest.warns(InfiniteLossWarning):
            history, grid_params = fitter.fit(observed, param_ranges, batch_size=20)

        self._check_history(history)
        self._check_grid_params(grid_params, params_copy, ["mu_x", "mu_y", "sigma"])
