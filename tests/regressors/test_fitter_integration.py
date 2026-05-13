"""Integration tests for regressors with fitters."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.fitters import GridFitter
from prfmodel.fitters import LeastSquaresFitter
from prfmodel.fitters import SGDFitter
from prfmodel.fitters.adapter import Adapter
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.regressors import AdditiveRegressors
from prfmodel.regressors import ConvolvedRegressors
from prfmodel.regressors import RegressorsList
from prfmodel.stimuli import PRFStimulus

_LOSS_ATOL = 1e-6
_SGD_NUM_STEPS = 3


@pytest.fixture
def stimulus():
    """2D bar pRF stimulus."""
    return load_2d_prf_bar_stimulus()


@pytest.fixture
def impulse_model():
    """Impulse model instance."""
    return DerivativeTwoGammaImpulse()


@pytest.fixture
def model(impulse_model: DerivativeTwoGammaImpulse):
    """Gaussian 2D pRF model with an additive regressors_model."""
    return Gaussian2DPRFModel(
        impulse_model=impulse_model,
        regressors_model=RegressorsList(
            [
                AdditiveRegressors(names=["reg_add"]),
                ConvolvedRegressors(names=["reg_conv"], impulse_model=impulse_model),
            ],
        ),
    )


@pytest.fixture
def true_params():
    """Ground-truth parameters for one synthetic unit."""
    return pd.DataFrame(
        {
            "mu_x": [0.0],
            "mu_y": [0.0],
            "sigma": [1.0],
            "delay": [6.0],
            "dispersion": [0.9],
            "undershoot": [12.0],
            "u_dispersion": [0.9],
            "ratio": [0.48],
            "weight_deriv": [0.5],
            "baseline": [0.5],
            "amplitude": [2.0],
            "beta_reg_add": [1.5],
            "beta_reg_conv": [-0.75],
        },
    )


@pytest.fixture
def regressor_design(stimulus: PRFStimulus):
    """Random regressor design DataFrame for two motion columns."""
    rng = np.random.default_rng(2026)
    return pd.DataFrame(
        rng.standard_normal((stimulus.design.shape[0], 2)),
        columns=["reg_add", "reg_conv"],
    )


class TestLeastSquaresIntegration:
    """Verify LeastSquaresFitter recovers beta coefficients alongside the pRF amplitude."""

    def test_recover_betas(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        true_params: pd.DataFrame,
        regressor_design: pd.DataFrame,
    ):
        """Fit jointly for amplitude + betas and recover the ground-truth values."""
        observed = np.asarray(model(stimulus, true_params, regressors=regressor_design))

        init_params = true_params.copy()
        init_params["amplitude"] = 0.0
        init_params["baseline"] = 0.0
        init_params["beta_reg_add"] = 0.0
        init_params["beta_reg_conv"] = 0.0

        fitter = LeastSquaresFitter(model=model, stimulus=stimulus)
        history, fit_params = fitter.fit(
            observed,
            init_params,
            slope_name=["amplitude", "beta_reg_add", "beta_reg_conv"],
            intercept_name="baseline",
            regressors=regressor_design,
        )

        assert history.history["loss"][0] < _LOSS_ATOL
        for name in ("amplitude", "baseline", "beta_reg_add", "beta_reg_conv"):
            assert np.isclose(fit_params[name].iloc[0], true_params[name].iloc[0], atol=1e-4)

    def test_missing_regressors_raises(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        true_params: pd.DataFrame,
    ):
        """Forgetting to pass regressors when fitting a regressor model raises."""
        fitter = LeastSquaresFitter(model=model, stimulus=stimulus)
        observed = np.zeros((1, stimulus.design.shape[0]))
        with pytest.raises(ValueError, match="regressors_model"):
            fitter.fit(
                observed,
                true_params,
                slope_name="amplitude",
            )


class TestGridIntegration:
    """Smoke test for GridFitter with a regressors_model."""

    def test_grid_smoke(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        true_params: pd.DataFrame,
        regressor_design: pd.DataFrame,
    ):
        """GridFitter runs and returns finite losses when regressors are supplied."""
        observed = np.asarray(model(stimulus, true_params, regressors=regressor_design))

        parameter_values: dict = {
            name: np.array([float(true_params[name].iloc[0])])
            for name in [
                "mu_x",
                "mu_y",
                "sigma",
                "delay",
                "dispersion",
                "undershoot",
                "u_dispersion",
                "ratio",
                "weight_deriv",
                "baseline",
                "amplitude",
                "beta_reg_add",
                "beta_reg_conv",
            ]
        }

        fitter = GridFitter(model=model, stimulus=stimulus)
        history, grid_params = fitter.fit(observed, parameter_values, regressors=regressor_design)

        assert np.isfinite(history.history["loss"]).all()
        assert grid_params.shape == (1, len(parameter_values))


class TestSGDIntegration:
    """Smoke test for SGDFitter with a regressors_model (3 steps)."""

    def test_sgd_smoke(
        self,
        stimulus: PRFStimulus,
        model: Gaussian2DPRFModel,
        true_params: pd.DataFrame,
        regressor_design: pd.DataFrame,
    ):
        """SGDFitter runs without raising when regressors are supplied."""
        observed = np.asarray(model(stimulus, true_params, regressors=regressor_design))

        fitter = SGDFitter(model=model, stimulus=stimulus, adapter=Adapter())
        history, sgd_params = fitter.fit(
            observed,
            true_params,
            num_steps=_SGD_NUM_STEPS,
            regressors=regressor_design,
        )

        assert sgd_params.shape == true_params.shape
        assert len(history.history.get("loss", [])) == _SGD_NUM_STEPS
