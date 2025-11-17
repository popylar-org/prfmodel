"""Tests for stochastic gradient descent fitting."""

import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.adapter import Adapter
from prfmodel.adapter import ParameterTransform
from prfmodel.fitters.sgd import SGDFitter
from prfmodel.fitters.sgd import SGDHistory
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .conftest import TestSetup
from .conftest import parametrize_dtype


@parametrize_dtype
class TestSGDFitter(TestSetup):
    """Tests for SGDFitter class.

    Uses a `Gaussian2DPRFModel` model with a `keras.optimizers.Adam` optimizer and `keras.losses.MeanSquaredError` loss
    as a test case.

    """

    num_steps: int = 10

    def _check_history(self, history: SGDHistory) -> None:
        assert isinstance(history, SGDHistory)
        assert history.step == list(range(self.num_steps))
        assert isinstance(history.history, dict)
        assert all(isinstance(x, Tensor) for x in history.history["loss"])

    def _check_sgd_params(self, result_params: pd.DataFrame, params: pd.DataFrame) -> None:
        assert isinstance(result_params, pd.DataFrame)
        assert result_params.shape == params.shape

    @pytest.mark.parametrize(
        ("optimizer", "loss"),
        [(None, None), (keras.optimizers.Adam, keras.losses.MeanSquaredError)],
    )
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        optimizer: type[keras.optimizers.Optimizer],
        loss: type[keras.losses.Loss],
        params: pd.DataFrame,
        dtype: str,
    ):
        """Test that fit returns parameters with the correct shape."""
        # Instantiate class args if not None
        if optimizer is not None:
            optimizer = optimizer()

        if loss is not None:
            loss = loss()

        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            optimizer=optimizer,
            loss=loss,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, sgd_params = fitter.fit(observed, params, num_steps=self.num_steps)

        self._check_history(history)
        self._check_sgd_params(sgd_params, params)

    def test_fit_fixed_params(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        dtype: str,
    ):
        """Test that fit with fixed parameters returns parameters with the correct shape and fixed values."""
        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        fixed = ["baseline", "amplitude"]

        history, sgd_params = fitter.fit(observed, params, fixed_parameters=fixed, num_steps=self.num_steps)

        self._check_history(history)
        self._check_sgd_params(sgd_params, params)
        assert np.all(sgd_params[fixed] == params[fixed].astype(get_dtype(dtype)))

    def test_fit_adapter(
        self,
        stimulus: Stimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        dtype: str,
    ):
        """Test that fit with an adapter returns parameters with the correct shape."""
        adapter = Adapter(
            [
                ParameterTransform(["sigma", "shape_1"], keras.ops.log, keras.ops.exp),
            ],
        )

        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            adapter=adapter,
            dtype=dtype,
        )

        observed = model(stimulus, params)

        history, sgd_params = fitter.fit(observed, params, num_steps=self.num_steps)

        self._check_history(history)
        self._check_sgd_params(sgd_params, params)
