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


def _param_ranges() -> dict[str, list]:
    """Parameter ranges shared by the grid search test classes.

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
        """Parameter ranges."""
        return _param_ranges()

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


class TestGridFitterValidation(TestSetup):
    """Tests that the grid is validated before the search runs.

    A batch is evaluated through :meth:`~prfmodel.models.base.BaseCanonical.call`, which stays traceable
    and therefore validates nothing. The checks the public facade would have run are done before the
    loop instead, so they have to be asserted here rather than relying on the model to raise.

    """

    @pytest.fixture
    def param_ranges(self):
        """Parameter ranges."""
        return _param_ranges()

    @pytest.fixture
    def observed(self, stimulus: PRFStimulus):
        """Target data of the right shape.

        The values are irrelevant: every test in this class asserts the search raises before any
        combination is evaluated.

        """
        return np.zeros((3, stimulus.design.shape[0]))

    @pytest.mark.parametrize(
        "delay_range",
        [
            # The offending value sits on an axis shorter than the longest one, so it is only reached if
            # the axis is tiled out rather than truncated, and in either position within that axis.
            [-1.0, 6.0],
            [6.0, -1.0],
            [0.0],
        ],
    )
    def test_out_of_domain_parameter_raises(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        param_ranges: dict[str, np.ndarray],
        observed: np.ndarray,
        delay_range: list[float],
    ):
        """Test that a grid sweeping a parameter outside the model domain raises rather than scoring it.

        A non-positive `delay` leaves the gamma density undefined. Evaluated through `call` it yields a
        finite but meaningless value that is free to win the `argmin`, so it has to be rejected up front.

        """
        fitter = GridFitter(model=model, stimulus=stimulus)

        with pytest.raises(ValueError, match="must be > 0"):
            fitter.fit(observed, {**param_ranges, "delay": delay_range})

    def test_out_of_domain_value_on_the_longest_axis_raises(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        param_ranges: dict[str, np.ndarray],
        observed: np.ndarray,
    ):
        """Test that the offending value is still found when its axis is the one every other is tiled to."""
        fitter = GridFitter(model=model, stimulus=stimulus)
        longest = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0]

        with pytest.raises(ValueError, match="must be > 0"):
            fitter.fit(observed, {**param_ranges, "delay": longest})

    def test_missing_parameter_raises(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        param_ranges: dict[str, np.ndarray],
        observed: np.ndarray,
    ):
        """Test that a grid that omits a required parameter names it instead of failing on a lookup."""
        fitter = GridFitter(model=model, stimulus=stimulus)
        incomplete = {name: values for name, values in param_ranges.items() if name != "sigma"}

        with pytest.raises(ValueError, match=r"Missing required parameter names.*sigma"):
            fitter.fit(observed, incomplete)

    def test_regressors_without_a_regressors_model_raises(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        param_ranges: dict[str, np.ndarray],
        observed: np.ndarray,
    ):
        """Test that regressors supplied to a model without a regressors model raise rather than being ignored."""
        fitter = GridFitter(model=model, stimulus=stimulus)
        design = pd.DataFrame({"reg": np.zeros(stimulus.design.shape[0])})

        with pytest.raises(ValueError, match="regressors_model"):
            fitter.fit(observed, param_ranges, regressors=design)

    def test_valid_grid_does_not_raise(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
    ):
        """Test that the validation lets a grid the model is defined on through unchanged."""
        fitter = GridFitter(model=model, stimulus=stimulus)

        history, grid_params = fitter.fit(model(stimulus, params), param_ranges, batch_size=20)

        assert np.isfinite(history.history["loss"]).all()
        assert grid_params.shape == params.shape


class TestGridFitterCompiledStep(TestSetup):
    """Tests for compiling the evaluation of a parameter batch.

    Compiling changes how a batch is evaluated but must not change which parameter combination wins.
    The batch also has to stay traceable: any Python-level branch on a tensor value, or any tensor cached
    across calls, raises once the evaluation is compiled but not while it runs eagerly.

    """

    @pytest.fixture
    def param_ranges(self):
        """Parameter ranges."""
        return _param_ranges()

    def test_the_batch_is_evaluated_eagerly_by_default(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
    ):
        """Test that compilation is never switched on for the caller.

        Unlike an optimization step, a parameter batch is evaluated once, so the trace is not amortized and
        whether compiling is faster depends on the grid size. That is not something the fitter can know.

        """
        fitter = GridFitter(model=model, stimulus=stimulus)

        assert not fitter.compile_step

    def test_compilation_can_be_switched_on(self, stimulus: PRFStimulus, model: Gaussian2DPRFModel):
        """Test that every backend has a compilation primitive available."""
        assert GridFitter(model=model, stimulus=stimulus, compile_step=True).compile_step

    @pytest.mark.parametrize("batch_size", [20, 7])
    def test_compiled_and_eager_searches_agree(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
        batch_size: int,
    ):
        """Test that compiling the batch does not change the combination the search arrives at.

        A `batch_size` that does not divide the grid size leaves a smaller final batch, which is traced a
        second time. That second trace is the case in which a tensor cached during the first one would
        surface, so it is exercised alongside the evenly divided case.

        A mean squared error loss is used rather than the default
        :class:`~prfmodel.fitters.losses.CorrelationLoss`, which leaves `baseline` and `amplitude`
        non-identifiable: every one of their grid values then yields the same loss, and which one wins is
        an arbitrary tie-break that float reassociation is free to decide differently.

        """
        observed = model(stimulus, params)
        loss = keras.losses.MeanSquaredError(reduction="none")

        _, compiled = GridFitter(model=model, stimulus=stimulus, loss=loss, compile_step=True).fit(
            observed,
            param_ranges,
            batch_size=batch_size,
        )
        _, eager = GridFitter(model=model, stimulus=stimulus, loss=loss, compile_step=False).fit(
            observed,
            param_ranges,
            batch_size=batch_size,
        )

        pd.testing.assert_frame_equal(compiled, eager)

    def test_ragged_final_batch_recovers_the_data_generating_parameters(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        param_ranges: dict[str, np.ndarray],
    ):
        """Test that a smaller final batch does not let a stale trace win the search.

        The final batch is not padded, because a padded combination could win the `argmin` and be
        returned as the estimate. This asserts the true parameters are still recovered exactly.

        """
        fitter = GridFitter(model=model, stimulus=stimulus, compile_step=True)

        _, grid_params = fitter.fit(model(stimulus, params), param_ranges, batch_size=7)

        check_params = ["mu_x", "mu_y", "sigma"]
        assert np.allclose(grid_params[check_params], params[check_params])
