"""Tests for linear fitting."""

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.dataframe_regression import DataFrameRegressionFixture
from prfmodel.fitters.linear import LeastSquaresFitter
from prfmodel.fitters.linear import LeastSquaresHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus
from .conftest import TestSetup
from .conftest import parametrize_dtype
from .conftest import parametrize_impulse_model


class TestLeastSquaresFitter(TestSetup):
    """Tests for GridFitter class."""

    def _check_history(self, history: LeastSquaresHistory) -> None:
        assert isinstance(history, LeastSquaresHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)

    def _check_least_squares_params(self, result_params: pd.DataFrame, params: pd.DataFrame) -> None:
        assert isinstance(result_params, pd.DataFrame)
        assert result_params.shape == params.shape

    @pytest.mark.parametrize("target_parameters", [[], ["a", "b", "c"]])
    def test_fit_target_parameters_value_error(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        target_parameters: tuple[str],
    ):
        """Test that 'target_parameters' with incorrect length raises error."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
        )

        observed = model(stimulus, params)

        with pytest.raises(ValueError):
            _ = fitter.fit(observed, params, target_parameters=target_parameters)

    @pytest.mark.parametrize("target_parameters", [None, ("a", "b", "c")])
    def test_fit_target_parameters_type_error(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        target_parameters: tuple[str],
    ):
        """Test that 'target_parameters' with incorrect input type raises error."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
        )

        observed = model(stimulus, params)

        with pytest.raises(TypeError):
            _ = fitter.fit(observed, params, target_parameters=target_parameters)

    @parametrize_dtype
    @parametrize_impulse_model
    @pytest.mark.parametrize("target_parameters", [["amplitude"], ["baseline"], ["amplitude", "baseline"]])
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        dataframe_regression: DataFrameRegressionFixture,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        target_parameters: tuple[str],
        dtype: str,
    ):
        """Test that fit returns objects with the correct type and attributes."""
        fitter = LeastSquaresFitter(
            model=model,
            stimulus=stimulus,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, ls_params = fitter.fit(observed, params, target_parameters=target_parameters)

        self._check_history(history)
        self._check_least_squares_params(ls_params, params)

        dataframe_regression.check(ls_params, default_tolerance={"atol": 1e-6})
