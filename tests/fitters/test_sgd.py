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
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import ShiftedGammaImpulse
from prfmodel.impulse import TwoGammaImpulse
from prfmodel.models.prf import DivNormGaussian2DPRFModel
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from tests.conftest import PRFStimulusSetup
from tests.conftest import TestSetup
from tests.conftest import parametrize_impulse_model
from .conftest import parametrize_dtype
from .conftest import skip_torch
from .conftest import skip_windows

_ATOL = 1e-3


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

    def _check_params_dtype(self, result_params: pd.DataFrame, param_names: list[str], dtype: str | None) -> None:
        dtype = get_dtype(dtype)
        for name in param_names:
            assert result_params[name].dtype == dtype

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
        self._check_params_dtype(sgd_params, sgd_params.columns, dtype)
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
        self._check_params_dtype(sgd_params, sgd_params.columns, dtype)

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
        self._check_params_dtype(sgd_params, sgd_params.columns, dtype)
        self._check_no_gradient_warnings(record)
        self._check_params_moved(
            sgd_params,
            params,
            moving_params=[c for c in params.columns if c not in (fixed_parameters or [])],
        )

    def test_extra_adapter_columns_are_nontrainable(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        dtype: str,
    ):
        """A label column becomes no optimization variable and is carried through to the estimates."""
        params_extra = params.assign(min_sigma=[0.05] * len(params))
        observed = np.asarray(model(stimulus, params))

        adapter = Adapter([ParameterConstraint(["sigma"], lower="min_sigma")])

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, fit_params = SGDFitter(model=model, stimulus=stimulus, adapter=adapter, dtype=dtype).fit(
                observed,
                params_extra,
                num_steps=self.num_steps,
            )

        assert list(fit_params.columns) == list(params_extra.columns)
        assert np.all(fit_params["min_sigma"] == params_extra["min_sigma"].astype(get_dtype(dtype)))

        self._check_no_gradient_warnings(record)
        self._check_params_moved(fit_params, params, moving_params=[c for c in params.columns if c != "min_sigma"])

    @pytest.mark.heavy
    def test_fit_batch_size(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        true_params: pd.DataFrame,
        dtype: str,
    ):
        """Test that fitting with batch_size produces the same final parameters as fitting all at once.

        Uses more optimization steps than other tests in this file: with very few steps, the comparison is
        dominated by floating-point summation-order noise between differently-shaped batches (the trajectories
        have not converged yet, so tiny per-step differences compound). With enough steps to approach
        convergence, that noise vanishes, which is what this test actually needs to verify.

        Unlike `LeastSquaresFitter`, the step-wise loss history is not directly comparable between the full and
        batched runs: a step's loss is an aggregate over whichever units are in that batch, so it means something
        different depending on how units were grouped. What batching should preserve is the step *bookkeeping*
        (steps keep counting up across batches) and the final per-unit parameters (checked below).

        """
        num_steps = 200
        num_batches = 3
        fixed_parameters = ["delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv"]

        fitter = SGDFitter(model=model, stimulus=stimulus, dtype=dtype)

        observed = model(stimulus, true_params)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            history_full, params_full = fitter.fit(
                observed,
                params,
                num_steps=num_steps,
                fixed_parameters=fixed_parameters,
            )
            history_batched, params_batched = fitter.fit(
                observed,
                params,
                num_steps=num_steps,
                batch_size=1,
                fixed_parameters=fixed_parameters,
            )

        assert history_full.step == list(range(num_steps))
        assert history_batched.step == list(range(num_batches * num_steps))
        self._check_sgd_params_shape(params_batched, params)
        self._check_no_gradient_warnings(record)
        self._check_params_moved(
            params_batched,
            params,
            moving_params=[c for c in params.columns if c not in (fixed_parameters or [])],
        )

        pd.testing.assert_frame_equal(params_full, params_batched, atol=_ATOL)


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
        """Return starting parameters for DoG model passed to the fitter."""
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
        """Data-generating parameters for DoG model."""
        true_params = dog_init_params.copy()
        true_params["mu_x"] += 0.5
        true_params["mu_y"] -= 0.5
        true_params["sigma_center"] += 0.5
        true_params["sigma_surround"] += 0.5
        return true_params

    @pytest.fixture
    def dog_moving_params(self) -> list[str]:
        """Return moving parameters for DoG model."""
        return ["mu_x", "mu_y", "sigma_center", "sigma_surround"]

    @pytest.fixture
    def div_norm_model(self) -> DivNormGaussian2DPRFModel:
        """DivNorm (divisive normalization) canonical pRF model."""
        return DivNormGaussian2DPRFModel()

    @pytest.fixture
    def div_norm_init_params(self) -> pd.DataFrame:
        """Return starting parameters for DivNorm model passed to the fitter."""
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
        """Data-generating parameters for DivNorm model."""
        true_params = div_norm_init_params.copy()
        true_params["mu_x"] += 0.5
        true_params["mu_y"] -= 0.5
        true_params["sigma_activation"] += 0.5
        true_params["sigma_normalization"] += 0.5
        return true_params

    @pytest.fixture
    def div_norm_moving_params(self) -> list[str]:
        """Return moving parameters for DivNorm model."""
        return ["mu_x", "mu_y", "sigma_activation", "sigma_normalization"]

    def test_sgd_moves_params_dog(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: PRFStimulus,
        dog_model: DoG2DPRFModel,
        dog_init_params: pd.DataFrame,
        dog_true_params: pd.DataFrame,
        dog_moving_params: list[str],
        dtype: str,
    ):
        """Test that SGD udpates moving parameters for DoG model."""
        fitter = SGDFitter(model=dog_model, stimulus=stimulus, dtype=dtype)

        observed = dog_model(stimulus, dog_true_params)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, sgd_params = fitter.fit(observed, dog_init_params, num_steps=self.num_steps)

        self._check_no_gradient_warnings(record)
        self._check_params_moved(sgd_params, dog_init_params, dog_moving_params)

    def test_sgd_moves_params_div_norm(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        stimulus: PRFStimulus,
        div_norm_model: DivNormGaussian2DPRFModel,
        div_norm_init_params: pd.DataFrame,
        div_norm_true_params: pd.DataFrame,
        div_norm_moving_params: list[str],
        dtype: str,
    ):
        """Test that SGD udpates moving parameters for DivNorm model."""
        fitter = SGDFitter(model=div_norm_model, stimulus=stimulus, dtype=dtype)

        observed = div_norm_model(stimulus, div_norm_true_params)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _, sgd_params = fitter.fit(observed, div_norm_init_params, num_steps=self.num_steps)

        self._check_no_gradient_warnings(record)
        self._check_params_moved(sgd_params, div_norm_init_params, div_norm_moving_params)


class TestSGDFitterConstraint(PRFStimulusSetup):
    """Tests that a `ParameterConstraint` actually binds the parameter the model sees.

    The constraint is enforced by the inverse transform applied before every prediction, so what
    matters is the parameter on the natural scale, not the value of the underlying variable. These
    tests drive the optimizer hard enough to push against the bound and check that it never gives.

    Only the pRF response parameters are fitted, so that the behaviour under test is the constraint
    itself rather than some other parameter diverging.

    """

    num_steps: int = 60
    lower_bound: float = 0.75

    @pytest.fixture
    def response_model(self):
        """Gaussian pRF model with no impulse or scaling stage."""
        return Gaussian2DPRFModel(impulse_model=None, scaling_model=None)

    @pytest.fixture
    def response_params(self):
        """Parameters for the response-only model."""
        return pd.DataFrame({"mu_x": [-1.0, 1.0], "mu_y": [1.0, -1.0], "sigma": [1.5, 2.0]})

    def test_constraint_holds_when_optimizer_pushes_against_bound(
        self,
        stimulus: PRFStimulus,
        response_model: Gaussian2DPRFModel,
        response_params: pd.DataFrame,
    ):
        """Test that a lower-bounded parameter stays above its bound and never becomes NaN.

        `sigma` is started just inside a bound above its true value, so the descent direction points
        straight at the bound, and a large learning rate makes the optimizer overshoot it.

        """
        adapter = Adapter([ParameterConstraint(["sigma"], lower=self.lower_bound)])

        init_params = response_params.copy()
        init_params["sigma"] = self.lower_bound + 0.05

        fitter = SGDFitter(
            model=response_model,
            stimulus=stimulus,
            adapter=adapter,
            optimizer=keras.optimizers.Adam(learning_rate=0.5),
        )

        observed = response_model(stimulus, response_params)

        _, sgd_params = fitter.fit(observed, init_params, num_steps=self.num_steps)

        sigma = sgd_params["sigma"].to_numpy()

        assert np.all(np.isfinite(sigma)), f"Constrained parameter became non-finite: {sigma}"
        assert np.all(sigma >= self.lower_bound), (
            f"Constrained parameter fell below its lower bound {self.lower_bound}: {sigma}"
        )

    def test_constraint_round_trips_through_the_fitter(
        self,
        stimulus: PRFStimulus,
        response_model: Gaussian2DPRFModel,
        response_params: pd.DataFrame,
    ):
        """Test that zero optimization steps return the starting values on the natural scale.

        A direction error in the transform pair survives a round trip in isolation, but would show up
        here as returned parameters that differ from the ones supplied.

        """
        adapter = Adapter([ParameterConstraint(["sigma"], lower=self.lower_bound)])

        fitter = SGDFitter(model=response_model, stimulus=stimulus, adapter=adapter)

        observed = response_model(stimulus, response_params)

        _, sgd_params = fitter.fit(observed, response_params, num_steps=0)

        np.testing.assert_allclose(
            sgd_params["sigma"].to_numpy(),
            response_params["sigma"].to_numpy(),
            rtol=1e-5,
            err_msg="Constrained parameter did not survive a round trip through the fitter",
        )


class TestSGDFitterCompiledStep(PRFStimulusSetup):
    """Tests for compiling the optimization step.

    Compiling changes how the step executes but must not change what it computes. The step also has to
    stay traceable: any Python-level branch on a tensor value, or any tensor used where the backend
    expects a Python integer, raises once the step is compiled but not while it runs eagerly.

    """

    num_steps: int = 10

    @pytest.fixture
    def response_model(self):
        """Gaussian pRF model with no impulse or scaling stage."""
        return Gaussian2DPRFModel(impulse_model=None, scaling_model=None)

    @pytest.fixture
    def init_params(self):
        """Return starting values displaced from the values used to simulate the data."""
        return pd.DataFrame({"mu_y": [0.5, -0.5], "mu_x": [-0.5, 0.5], "sigma": [1.0, 1.0]})

    @pytest.fixture
    def true_params(self):
        """Parameters used to simulate the target data."""
        return pd.DataFrame({"mu_y": [1.0, -1.0], "mu_x": [-1.0, 1.0], "sigma": [1.5, 2.0]})

    def _fit(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        init_params: pd.DataFrame,
        true_params: pd.DataFrame,
        compile_step: bool,
    ) -> pd.DataFrame:
        keras.utils.set_random_seed(0)
        fitter = SGDFitter(
            model=model,
            stimulus=stimulus,
            optimizer=keras.optimizers.Adam(learning_rate=0.1),
            compile_step=compile_step,
        )
        _, params = fitter.fit(model(stimulus, true_params), init_params, num_steps=self.num_steps)
        return params

    def test_the_step_runs_eagerly_by_default(self, stimulus: PRFStimulus, response_model: Gaussian2DPRFModel):
        """Test that compilation is never switched on for the caller.

        The eager path is the one that is debuggable, and whether compiling is faster depends on the backend
        and the problem size, so it has to be asked for.

        """
        fitter = SGDFitter(model=response_model, stimulus=stimulus)

        assert not fitter.compile_step

    def test_compilation_can_be_switched_on(self, stimulus: PRFStimulus, response_model: Gaussian2DPRFModel):
        """Test that every backend has a compilation primitive available."""
        assert SGDFitter(model=response_model, stimulus=stimulus, compile_step=True).compile_step

    def test_compiled_and_eager_fits_agree(
        self,
        stimulus: PRFStimulus,
        response_model: Gaussian2DPRFModel,
        init_params: pd.DataFrame,
        true_params: pd.DataFrame,
    ):
        """Test that compiling the step does not change the parameters it arrives at.

        This is the assertion that makes compilation safe to switch on. The two paths run the same
        arithmetic in a different execution mode, so they agree to within float32 reassociation rather
        than exactly.

        """
        compiled = self._fit(stimulus, response_model, init_params, true_params, compile_step=True)
        eager = self._fit(stimulus, response_model, init_params, true_params, compile_step=False)

        for param in init_params.columns:
            np.testing.assert_allclose(
                compiled[param].to_numpy(),
                eager[param].to_numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Compiling the step changed the fitted value of {param!r}",
            )

    @pytest.mark.parametrize(
        "impulse_model",
        [DerivativeTwoGammaImpulse, ShiftedGammaImpulse, TwoGammaImpulse],
    )
    def test_full_pipeline_stays_traceable(self, stimulus: PRFStimulus, impulse_model: type):
        """Test that a fit with each impulse model runs with the step compiled.

        The impulse stage is where the density validation lives, and the encoding stage is where the
        reduction axes are built. Both have raised under tracing in the past, and neither shows up in an
        eager fit, so this runs the whole pipeline through the compiled path.

        """
        model = Gaussian2DPRFModel(impulse_model=impulse_model)

        params = pd.DataFrame({"mu_y": [1.0], "mu_x": [-1.0], "sigma": [1.5], "baseline": [0.0], "amplitude": [1.0]})
        params = params.assign(**{name: 0.5 for name in model.parameter_names if name not in params.columns})

        fitter = SGDFitter(model=model, stimulus=stimulus, compile_step=True)

        _, fitted = fitter.fit(model(stimulus, params), params, num_steps=2)

        assert np.all(np.isfinite(fitted.to_numpy())), f"Fit produced non-finite parameters: {fitted}"

    @pytest.mark.parametrize(
        "impulse_model",
        [DerivativeTwoGammaImpulse, ShiftedGammaImpulse, TwoGammaImpulse],
    )
    def test_a_negative_impulse_offset_fits_to_finite_parameters(
        self,
        stimulus: PRFStimulus,
        impulse_model: type,
    ):
        """Test that frames outside the density's support do not poison the gradient.

        A negative `offset` puts frames at or below zero on the time axis. The densities mask those to
        zero with `ops.where`, which propagates the gradient of *both* branches -- so the masked branch
        has to be evaluated on a substituted value rather than on the raw one, or the gradient would be
        NaN even though the forward value is fine. Only a gradient reveals it, so this fits rather than
        merely predicting.

        The impulse parameters are supplied explicitly rather than left to `default_parameters`, because
        a defaulted parameter is a constant that no gradient reaches -- which is exactly the case that
        would not exercise this.

        """
        impulse = impulse_model(offset=-5.0)
        model = Gaussian2DPRFModel(impulse_model=impulse)

        params = pd.DataFrame({"mu_y": [1.0], "mu_x": [-1.0], "sigma": [1.5], "baseline": [0.0], "amplitude": [1.0]})
        impulse_names = impulse.parameter_names
        params = params.assign(**dict.fromkeys(impulse_names, 1.5))
        params = params.assign(**{name: 0.5 for name in model.parameter_names if name not in params.columns})

        fitter = SGDFitter(model=model, stimulus=stimulus, compile_step=True)

        _, fitted = fitter.fit(model(stimulus, params), params, num_steps=5)

        assert np.all(np.isfinite(fitted.to_numpy())), f"Fit produced non-finite parameters: {fitted}"

    def test_invalid_starting_values_are_reported(self, stimulus: PRFStimulus):
        """Test that starting values outside a model's domain raise before the loop starts.

        Value validation skips itself under a trace so the step can be compiled, which would otherwise
        turn this mistake into a silent NaN loss. The fitter makes one prediction before anything is
        traced, so the model's own error message still reaches the user. This runs with the step compiled,
        because that is the case in which the pre-loop prediction is the only thing that reports it.

        """
        model = Gaussian2DPRFModel()

        params = pd.DataFrame({"mu_y": [1.0], "mu_x": [-1.0], "sigma": [1.5], "baseline": [0.0], "amplitude": [1.0]})
        params = params.assign(**{name: 0.5 for name in model.parameter_names if name not in params.columns})
        # A negative dispersion is outside the impulse model's domain, which 'BaseImpulse'
        # reports from its facade via the composite model's '_check_parameter_values' fan-out
        params["dispersion"] = -1.0

        fitter = SGDFitter(model=model, stimulus=stimulus, compile_step=True)

        with pytest.raises(ValueError, match="must be > 0"):
            fitter.fit(np.zeros((1, stimulus.design.shape[0])), params, num_steps=5)


class TestSGDFitterIgnoresExtraColumns(TestSetup):
    """Tests that a frame column no model reads passes through the fitter untouched."""

    num_steps: int = 2

    def test_ignores_extra_columns(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
    ):
        """A label column becomes no optimization variable and is carried through to the estimates."""
        labelled = params.assign(roi=["V1"] * len(params))
        observed = np.asarray(model(stimulus, params))

        _, fit_params = SGDFitter(model=model, stimulus=stimulus, adapter=Adapter()).fit(
            observed,
            labelled,
            num_steps=self.num_steps,
        )

        assert list(fit_params.columns) == list(labelled.columns)
        assert list(fit_params["roi"]) == list(labelled["roi"])
