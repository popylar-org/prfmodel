"""Tests for linear fitting."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.fitters import LeastSquaresFitter
from prfmodel.fitters import LeastSquaresHistory
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import Gaussian2DCSTPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup
from tests.conftest import TestSetup
from tests.conftest import parametrize_impulse_model
from tests.reference import _oracle
from .conftest import parametrize_dtype
from .conftest import skip_torch
from .conftest import skip_windows

_ATOL = 1e-3


class TestLeastSquaresFitter(TestSetup):
    """Tests for GridFitter class."""

    def _check_history(self, history: LeastSquaresHistory) -> None:
        assert isinstance(history, LeastSquaresHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)

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
    def test_fit(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        params: pd.DataFrame,
        intercept_name: str | None,
        dtype: str,
    ):
        """Test that the fitter recovers amplitude (and baseline) from independently generated data.

        The target comes from `tests.reference._oracle` rather than from `model`. With a
        self-generated target the recovery is circular -- the least-squares basis is built from the
        same model that produced the data, so the true coefficients sit at the optimum whether or not
        the forward model is right. Generating the target independently means recovering `amplitude`
        is evidence that the basis is correct, not only that `lstsq` works.

        """
        # When no intercept is fit, the model retains the baseline from the input parameters,
        # so any non-zero baseline gets folded into the basis and the slope cannot recover.
        true_params = params.copy()
        if intercept_name is None:
            true_params["baseline"] = 0.0

        # Give each unit its own HRF. The shared fixture uses one delay for all units, which hides any
        # defect that pairs a unit with another unit's impulse response -- the basis would still be
        # built from the right kernel by accident. These are fixed, not fitted, so varying them costs
        # nothing here.
        true_params["delay"] = [5.0, 6.0, 7.0]

        fitter = LeastSquaresFitter(model=model, stimulus=stimulus, dtype=dtype)

        observed = _oracle.predict(stimulus, true_params, frames=_oracle.impulse_frames())

        history, ls_params = fitter.fit(observed, true_params, slope_name="amplitude", intercept_name=intercept_name)

        self._check_history(history)

        np.testing.assert_allclose(
            ls_params["amplitude"].to_numpy(),
            true_params["amplitude"].to_numpy(),
            atol=_ATOL,
        )
        if intercept_name is not None:
            np.testing.assert_allclose(
                ls_params[intercept_name].to_numpy(),
                true_params[intercept_name].to_numpy(),
                atol=_ATOL,
            )

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

        pd.testing.assert_frame_equal(params_full, params_batched, atol=_ATOL)
        np.testing.assert_allclose(history_full.history["loss"], history_batched.history["loss"], atol=_ATOL)


class TestLeastSquaresFitterMultiSlope(PRFStimulusSetup):
    """Tests for LeastSquaresFitter with multiple slope parameters."""

    @pytest.fixture
    def dog_model(self):
        """DoG 2D pRF model instance."""
        return DoG2DPRFModel()

    @pytest.fixture
    def dog_params(self):
        """Parameters dataframe for DoG model."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma_center": [0.5, 0.5, 0.5],
                "sigma_surround": [8.0, 8.0, 8.0],
                "delay": [6.0, 6.0, 6.0],
                "dispersion": [0.9, 0.9, 0.9],
                "undershoot": [12.0, 12.0, 12.0],
                "u_dispersion": [0.9, 0.9, 0.9],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "amplitude_center": [1.1, 1.0, 0.9],
                "amplitude_surround": [0.5, 0.3, 0.1],
                "baseline": [0.0, 0.1, 0.2],
            },
        )

    @skip_windows
    @skip_torch
    @pytest.mark.parametrize("intercept_name", [None, "baseline"])
    def test_fit_multi_slope(
        self,
        stimulus: PRFStimulus,
        dog_model: DoG2DPRFModel,
        dog_params: pd.DataFrame,
        intercept_name: str | None,
    ):
        """Test that the fitter recovers both DoG amplitudes (and baseline) from independent data.

        As in `test_fit`, the target is generated by `tests.reference._oracle` so that recovering the
        two amplitudes says something about the centre and surround bases rather than only about the
        solver. This is the stronger of the two cases: with two slopes the basis has to be right in
        two independent directions for both coefficients to come back.

        """
        true_params = dog_params.copy()
        if intercept_name is None:
            true_params["baseline"] = 0.0

        # See `TestLeastSquaresFitter.test_fit`: one HRF per unit, so that a unit paired with another
        # unit's impulse response cannot pass unnoticed.
        true_params["delay"] = [5.0, 6.0, 7.0]

        fitter = LeastSquaresFitter(model=dog_model, stimulus=stimulus)

        observed = _oracle.predict(stimulus, true_params, frames=_oracle.impulse_frames(), surround=True)

        history, ls_params = fitter.fit(
            observed,
            true_params,
            slope_name=["amplitude_center", "amplitude_surround"],
            intercept_name=intercept_name,
        )

        assert isinstance(history, LeastSquaresHistory)
        assert isinstance(history.history, dict)
        assert isinstance(history.history["loss"], np.ndarray)
        assert isinstance(ls_params, pd.DataFrame)
        assert ls_params.shape == true_params.shape

        for name in ("amplitude_center", "amplitude_surround"):
            np.testing.assert_allclose(
                ls_params[name].to_numpy(),
                true_params[name].to_numpy(),
                atol=_ATOL,
            )
        if intercept_name is not None:
            np.testing.assert_allclose(
                ls_params[intercept_name].to_numpy(),
                true_params[intercept_name].to_numpy(),
                atol=_ATOL,
            )

    @skip_windows
    @skip_torch
    @pytest.mark.parametrize("intercept_name", [None, "baseline"])
    def test_fit_multi_slope_batch_size(
        self,
        stimulus: PRFStimulus,
        dog_model: DoG2DPRFModel,
        dog_params: pd.DataFrame,
        intercept_name: str | None,
    ):
        """Test that multi-slope fitting with batch_size produces the same results as fitting all at once."""
        fitter = LeastSquaresFitter(
            model=dog_model,
            stimulus=stimulus,
        )

        observed = dog_model(stimulus, dog_params)

        history_full, params_full = fitter.fit(
            observed,
            dog_params,
            slope_name=["amplitude_center", "amplitude_surround"],
            intercept_name=intercept_name,
        )
        history_batched, params_batched = fitter.fit(
            observed,
            dog_params,
            slope_name=["amplitude_center", "amplitude_surround"],
            intercept_name=intercept_name,
            batch_size=1,
        )

        pd.testing.assert_frame_equal(params_full, params_batched, atol=_ATOL)
        np.testing.assert_allclose(history_full.history["loss"], history_batched.history["loss"], atol=_ATOL)


class TestLeastSquaresFitterCST(PRFStimulusSetup):
    """Tests for LeastSquaresFitter with the compressive spatiotemporal model's channel weights.

    The reference estimates the sustained and transient scaling factors "in each voxel ... using a GLM
    approach". Those weights enter linearly once the spatial, timing and compression parameters are held
    fixed, so this fitter is where that step of the reference's procedure lives.

    """

    @pytest.fixture
    def cst_model(self):
        """CST 2D pRF model instance."""
        return Gaussian2DCSTPRFModel()

    @pytest.fixture
    def cst_params(self):
        """Parameters dataframe for the CST model, with per-unit weights to be recovered."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, -1.0],
                "mu_y": [1.0, 0.0, 0.5],
                "sigma": [1.0, 1.5, 2.0],
                "time_to_peak": [4.0, 5.0, 4.0],
                "n": [0.5, 0.8, 0.6],
                # Per-unit impulse responses, so a unit paired with another unit's HRF cannot pass unnoticed
                "delay": [5.0, 6.0, 7.0],
                "dispersion": [0.9, 0.9, 0.9],
                "undershoot": [12.0, 12.0, 12.0],
                "u_dispersion": [0.9, 0.9, 0.9],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "beta_sustained": [2.0, -1.5, 0.7],
                "beta_transient": [-0.5, 3.0, 1.4],
                "baseline": [0.3, -0.2, 0.0],
            },
        )

    @skip_windows
    @skip_torch
    @pytest.mark.parametrize("intercept_name", [None, "baseline"])
    def test_fit_recovers_channel_weights(
        self,
        stimulus: PRFStimulus,
        cst_model: Gaussian2DCSTPRFModel,
        cst_params: pd.DataFrame,
        intercept_name: str | None,
    ):
        """Test that both channel weights (and the baseline) are recovered by the linear solve.

        Unlike the DoG case above there is no oracle for this model, so the target is generated by the
        model under test. That makes this a statement about the solver and about the two channel bases
        being linearly independent, not about the forward model, which is pinned separately by the
        reduction identities in `tests/reference/test_model_identities.py`. Independence is the part worth
        asserting: the sustained and transient channels share a receptive field and a timing parameter, so
        collinear bases would make the solve degenerate and the weights unrecoverable.

        Negative weights are included deliberately. The channels are rectified before they are weighted,
        so a sign error in the rectifier cannot be absorbed by a positive-only solution.

        """
        true_params = cst_params.copy()
        if intercept_name is None:
            true_params["baseline"] = 0.0

        observed = cst_model(stimulus, true_params)

        init_params = true_params.assign(beta_sustained=0.0, beta_transient=0.0, baseline=0.0)

        fitter = LeastSquaresFitter(model=cst_model, stimulus=stimulus)

        history, ls_params = fitter.fit(
            observed,
            init_params,
            slope_name=["beta_sustained", "beta_transient"],
            intercept_name=intercept_name,
        )

        assert isinstance(history, LeastSquaresHistory)
        assert isinstance(ls_params, pd.DataFrame)
        assert ls_params.shape == true_params.shape

        for name in ("beta_sustained", "beta_transient"):
            np.testing.assert_allclose(
                ls_params[name].to_numpy(),
                true_params[name].to_numpy(),
                atol=_ATOL,
            )
        if intercept_name is not None:
            np.testing.assert_allclose(
                ls_params[intercept_name].to_numpy(),
                true_params[intercept_name].to_numpy(),
                atol=_ATOL,
            )
