"""Tests for stochastic gradient descent fitting."""

import warnings
import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters import SGDFitter
from prfmodel.fitters import SGDHistory
from prfmodel.fitters.adapter import Adapter
from prfmodel.fitters.adapter import ParameterConstraint
from prfmodel.fitters.adapter import ParameterTransform
from prfmodel.models.prf import DivNormGaussian2DPRFModel
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from tests.conftest import TestSetup
from tests.conftest import parametrize_impulse_model
from .conftest import parametrize_dtype
from .conftest import skip_torch
from .conftest import skip_windows


class _SGDGradientChecks:
    """Shared assertions for SGD fitting tests, reused across model architectures.

    Both checks target the class of bug where a parameter is silently detached from the
    gradient tape (e.g. by round-tripping through a :class:`pandas.DataFrame`): Keras then warns
    that gradients don't exist for that variable, and the optimizer leaves it exactly at its
    starting value instead of moving it.
    """

    def _check_no_gradient_warnings(self, record: list[warnings.WarningMessage]) -> None:
        """Assert that fitting did not emit a 'Gradients do not exist' warning for any parameter."""
        gradient_warnings = [w.message for w in record if "Gradients do not exist" in str(w.message)]
        assert not gradient_warnings, f"Unexpected gradient warnings: {gradient_warnings}"

    def _check_params_moved(
        self,
        result_params: pd.DataFrame,
        init_params: pd.DataFrame,
        moving_params: list[str],
    ) -> None:
        """Assert that each parameter in ``moving_params`` changed value during fitting."""
        for param in moving_params:
            assert not np.allclose(
                result_params[param].to_numpy(),
                init_params[param].to_numpy(),
            ), f"{param!r} did not change during SGD fitting; gradients may not be flowing to it."


@skip_windows
@skip_torch
@parametrize_dtype
class TestSGDFitter(_SGDGradientChecks, TestSetup):
    """Tests for SGDFitter class.

    Uses a `Gaussian2DPRFModel` model with a `keras.optimizers.Adam` optimizer and `keras.losses.MeanSquaredError` loss
    as a test case.

    """

    num_steps: int = 10

    @pytest.fixture
    def true_params(self, params: pd.DataFrame) -> pd.DataFrame:
        """Data-generating parameters, offset from ``params`` so that fitting has genuine, nonzero gradients."""
        true_params = params.copy()
        true_params["mu_x"] += 0.3
        true_params["mu_y"] -= 0.3
        true_params["sigma"] += 0.3
        true_params["delay"] += 0.5
        true_params["dispersion"] += 0.05
        true_params["undershoot"] += 0.5
        true_params["u_dispersion"] += 0.05
        true_params["ratio"] += 0.02
        true_params["weight_deriv"] += 0.1
        true_params["baseline"] += 0.1
        true_params["amplitude"] += 0.1
        return true_params

    def _check_history(self, history: SGDHistory) -> None:
        assert isinstance(history, SGDHistory)
        assert history.step == list(range(self.num_steps))
        assert isinstance(history.history, dict)
        assert all(isinstance(x, Tensor) for x in history.history["loss"])

    def _check_sgd_params_shape(self, result_params: pd.DataFrame, params: pd.DataFrame) -> None:
        assert isinstance(result_params, pd.DataFrame)
        assert result_params.shape == params.shape

    @pytest.mark.parametrize(
        ("optimizer", "loss"),
        [(None, None), (keras.optimizers.Adam, keras.losses.MeanSquaredError)],
    )
    def test_fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        optimizer: type[keras.optimizers.Optimizer],
        loss: type[keras.losses.Loss],
        params: pd.DataFrame,
        true_params: pd.DataFrame,
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

        observed = model(stimulus, true_params)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            history, sgd_params = fitter.fit(observed, params, num_steps=self.num_steps)

        self._check_history(history)
        self._check_sgd_params_shape(sgd_params, params)
        self._check_no_gradient_warnings(record)
        self._check_params_moved(sgd_params, params, moving_params=list(params.columns))

    def test_fit_fixed_params(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        true_params: pd.DataFrame,
        dtype: str,
    ):
        """Test that fit with fixed parameters returns parameters with the correct shape and fixed values."""
        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            dtype=dtype,
        )

        observed = model(stimulus, true_params)

        fixed = ["baseline", "amplitude"]

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            history, sgd_params = fitter.fit(observed, params, fixed_parameters=fixed, num_steps=self.num_steps)

        self._check_history(history)
        self._check_sgd_params_shape(sgd_params, params)

        assert np.all(sgd_params[fixed] == params[fixed].astype(get_dtype(dtype)))

        self._check_no_gradient_warnings(record)
        self._check_params_moved(sgd_params, params, moving_params=[c for c in params.columns if c not in fixed])

    @parametrize_impulse_model
    def test_fit_adapter(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        true_params: pd.DataFrame,
        dtype: str,
    ):
        """Test that fit with an adapter returns parameters with the correct shape."""
        adapter = Adapter(
            [
                ParameterTransform(["sigma", "delay"], keras.ops.log, keras.ops.exp),
                ParameterConstraint(["delay"], lower="dispersion", bound_fun=keras.ops.log),
            ],
        )

        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            adapter=adapter,
            dtype=dtype,
        )

        observed = model(stimulus, true_params)

        fixed_parameters = None

        # We need to fix the default parameters of the impulse model because they won't have gradients
        if model.models["impulse_model"].default_parameters is not None:
            fixed_parameters = model.models["impulse_model"].default_parameters.keys()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            history, sgd_params = fitter.fit(
                observed,
                params,
                num_steps=self.num_steps,
                fixed_parameters=fixed_parameters,
            )

        self._check_history(history)
        self._check_sgd_params_shape(sgd_params, params)
        self._check_no_gradient_warnings(record)
        self._check_params_moved(
            sgd_params,
            params,
            moving_params=[c for c in params.columns if c not in (fixed_parameters or [])],
        )


@skip_windows
@skip_torch
@parametrize_dtype
class TestSGDDualResponse(_SGDGradientChecks, TestSetup):
    """Verify SGD updates shared and response-specific pRF parameters for DoG and DivNorm models."""

    num_steps: int = 10

    @pytest.fixture
    def dog_model(self) -> DoG2DPRFModel:
        """DoG (Difference of Gaussians) canonical pRF model."""
        return DoG2DPRFModel()

    @pytest.fixture
    def dog_init_params(self) -> pd.DataFrame:
        """Return starting parameters passed to the fitter."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0],
                "mu_y": [1.0, 0.0],
                "sigma_center": [1.0, 1.5],
                "sigma_surround": [2.0, 3.0],
                "delay": [6.0, 7.0],
                "dispersion": [0.9, 1.0],
                "undershoot": [12.0, 11.0],
                "u_dispersion": [0.9, 1.0],
                "ratio": [0.48, 0.48],
                "weight_deriv": [0.5, 0.5],
                "amplitude_center": [1.1, 1.0],
                "amplitude_surround": [0.5, 0.3],
                "baseline": [0.0, 0.1],
            },
        )

    @pytest.fixture
    def dog_true_params(self, dog_init_params: pd.DataFrame) -> pd.DataFrame:
        """Data-generating parameters, offset from ``dog_init_params`` in the mu/sigma columns."""
        true_params = dog_init_params.copy()
        true_params["mu_x"] += 0.5
        true_params["mu_y"] -= 0.5
        true_params["sigma_center"] += 0.5
        true_params["sigma_surround"] += 0.5
        return true_params

    @pytest.fixture
    def div_norm_model(self) -> DivNormGaussian2DPRFModel:
        """Divisive normalization canonical pRF model."""
        return DivNormGaussian2DPRFModel()

    @pytest.fixture
    def div_norm_init_params(self) -> pd.DataFrame:
        """Return starting parameters passed to the fitter."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0],
                "mu_y": [1.0, 0.0],
                "sigma_activation": [1.0, 1.5],
                "sigma_normalization": [2.0, 3.0],
                "weight_deriv": [-0.5, -0.5],
                "amplitude_activation": [1.1, 1.0],
                "baseline_activation": [0.0, 0.1],
                "amplitude_normalization": [10.0, 5.0],
                "baseline_normalization": [20.0, 10.0],
                "baseline": [-0.5, 0.5],
            },
        )

    @pytest.fixture
    def div_norm_true_params(self, div_norm_init_params: pd.DataFrame) -> pd.DataFrame:
        """Data-generating parameters, offset from ``div_norm_init_params`` in the mu/sigma columns."""
        true_params = div_norm_init_params.copy()
        true_params["mu_x"] += 0.5
        true_params["mu_y"] -= 0.5
        true_params["sigma_activation"] += 0.5
        true_params["sigma_normalization"] += 0.5
        return true_params

    @pytest.mark.parametrize(
        ("model_fixture", "init_params_fixture", "true_params_fixture", "moving_params"),
        [
            (
                "dog_model",
                "dog_init_params",
                "dog_true_params",
                ["mu_x", "mu_y", "sigma_center", "sigma_surround"],
            ),
            (
                "div_norm_model",
                "div_norm_init_params",
                "div_norm_true_params",
                ["mu_x", "mu_y", "sigma_activation", "sigma_normalization"],
            ),
        ],
    )
    def test_sgd_moves_shared_and_response_specific_params(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        request: pytest.FixtureRequest,
        stimulus: PRFStimulus,
        model_fixture: str,
        init_params_fixture: str,
        true_params_fixture: str,
        moving_params: list[str],
        dtype: str,
    ):
        """SGD must update mu_x/mu_y and both response-specific sigma parameters, not just amplitude/baseline."""
        model = request.getfixturevalue(model_fixture)
        init_params = request.getfixturevalue(init_params_fixture)
        true_params = request.getfixturevalue(true_params_fixture)

        fitter = SGDFitter(model=model, stimulus=stimulus, dtype=dtype)

        observed = model(stimulus, true_params)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, sgd_params = fitter.fit(observed, init_params, num_steps=self.num_steps)

        self._check_no_gradient_warnings(record)
        self._check_params_moved(sgd_params, init_params, moving_params)
