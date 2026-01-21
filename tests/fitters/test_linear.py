"""Tests for linear fitting."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters.linear import LeastSquaresFitter
from prfmodel.fitters.linear import LeastSquaresHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus
from .conftest import TestSetup
from .conftest import parametrize_dtype


class TestLeastSquaresFitter(TestSetup):
    """Tests for GridFitter class."""

    def _check_history(self, history: LeastSquaresHistory) -> None:
        assert isinstance(history, LeastSquaresHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)

    def _check_least_squares_params(self, result_params: pd.DataFrame, params: pd.DataFrame) -> None:
        assert isinstance(result_params, pd.DataFrame)
        assert result_params.shape == params.shape

    @pytest.mark.parametrize(
        ("slope_name", "intercept_name"),
        [(None, None), ("amplitude", "test"), ("test", "baseline")],
    )
    def test_fit_names_value_error(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        slope_name: str,
        intercept_name: str,
    ):
        """Test that slope and intercept names not in parameters raise an error."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
        )

        observed = model(stimulus, params)

        with pytest.raises(ValueError):
            _ = fitter.fit(observed, params, slope_name=slope_name, intercept_name=intercept_name)

    @parametrize_dtype
    @pytest.mark.parametrize("intercept_name", [None, "baseline"])
    def test_fit(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        intercept_name: str | None,
        dtype: str,
    ):
        """Test that fit returns objects with the correct type and attributes."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, ls_params = fitter.fit(observed, params, slope_name="amplitude", intercept_name=intercept_name)

        self._check_history(history)
        self._check_least_squares_params(ls_params, params)
