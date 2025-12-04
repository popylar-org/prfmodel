"""Tests for linear fitting."""

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.dataframe_regression import DataFrameRegressionFixture
from prfmodel.fitters.linear import LeastSquaresFitter
from prfmodel.fitters.linear import LeastSquaresHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import PRFStimulus
from tests.conftest import TestSetup
from tests.conftest import parametrize_impulse_model
from .conftest import parametrize_dtype
from .conftest import skip_torch
from .conftest import skip_windows


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
        stimulus: PRFStimulus,
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

    @skip_windows
    @skip_torch
    @parametrize_dtype
    @parametrize_impulse_model
    @pytest.mark.parametrize("intercept_name", [None, "baseline"])
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        dataframe_regression: DataFrameRegressionFixture,
        stimulus: PRFStimulus,
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

        if dtype != "float64":
            dataframe_regression.check(ls_params, default_tolerance={"atol": 1e-6})

    @skip_windows
    @skip_torch
    @parametrize_impulse_model
    @pytest.mark.parametrize("intercept_name", [None, "baseline"])
    def test_fit_batch_size(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        intercept_name: str | None,
    ):
        """Test that fitting with batch_size produces the same results as fitting all at once."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
        )

        observed = model(stimulus, params)

        history_full, params_full = fitter.fit(
            observed,
            params,
            slope_name="amplitude",
            intercept_name=intercept_name,
        )
        history_batched, params_batched = fitter.fit(
            observed,
            params,
            slope_name="amplitude",
            intercept_name=intercept_name,
            batch_size=1,
        )

        pd.testing.assert_frame_equal(params_full, params_batched, atol=1e-4)
        np.testing.assert_allclose(history_full.history["loss"], history_batched.history["loss"], atol=1e-4)
