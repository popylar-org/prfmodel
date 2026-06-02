"""Integration tests for regressors with canonical models."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.cf import GaussianCFModel
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.regressors import AdditiveRegressors
from prfmodel.regressors import ConvolvedRegressors
from prfmodel.stimuli import CFStimulus
from prfmodel.stimuli import PRFStimulus


class TestPRFIntegration:
    """Integration tests for regressors with Gaussian2DPRFModel."""

    num_units = 2

    @pytest.fixture
    def stimulus(self):
        """2D bar pRF stimulus."""
        return load_2d_prf_bar_stimulus()

    @pytest.fixture
    def impulse_model(self):
        """Shared impulse model instance."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def base_params(self):
        """Parameters dataframe without regressor betas."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0],
                "mu_y": [1.0, 0.0],
                "sigma": [1.0, 1.5],
                "delay": [6.0, 6.0],
                "dispersion": [0.9, 0.9],
                "undershoot": [12.0, 12.0],
                "u_dispersion": [0.9, 0.9],
                "ratio": [0.48, 0.48],
                "weight_deriv": [0.5, 0.5],
                "baseline": [0.1, -0.1],
                "amplitude": [-2.0, 1.2],
            },
        )

    def test_additive_only(
        self,
        stimulus: PRFStimulus,
        base_params: pd.DataFrame,
        impulse_model: DerivativeTwoGammaImpulse,
    ):
        """An additive regressors model only adds the regressor contribution to the baseline prediction."""
        num_frames = stimulus.design.shape[0]
        rng = np.random.default_rng(42)
        design = pd.DataFrame(
            rng.standard_normal((num_frames, 3)),
            columns=["mx", "my", "mz"],
        )
        regressors = AdditiveRegressors(names=["mx", "my", "mz"])

        baseline_model = Gaussian2DPRFModel(impulse_model=impulse_model)
        model = Gaussian2DPRFModel(impulse_model=impulse_model, regressors_model=regressors)

        params = base_params.copy()
        params["beta_mx"] = [1.0, -0.5]
        params["beta_my"] = [0.0, 0.2]
        params["beta_mz"] = [-1.0, 1.0]

        for name in regressors.parameter_names:
            assert name in model.parameter_names

        baseline_resp = np.asarray(baseline_model(stimulus, base_params))
        full_resp = np.asarray(model(stimulus, params, regressors=design))

        betas = params[regressors.parameter_names].to_numpy()
        expected_contribution = betas @ design[["mx", "my", "mz"]].to_numpy().T

        assert full_resp.shape == baseline_resp.shape
        assert np.allclose(full_resp - baseline_resp, expected_contribution, atol=1e-4)

    def test_combined_dataframe_for_list(
        self,
        stimulus: PRFStimulus,
        base_params: pd.DataFrame,
        impulse_model: DerivativeTwoGammaImpulse,
    ):
        """A single combined confounds DataFrame works for a RegressorsList (broadcast to children)."""
        num_frames = stimulus.design.shape[0]
        rng = np.random.default_rng(123)
        confounds = pd.DataFrame(
            {
                "task": np.zeros(num_frames),
                "nuisance_a": rng.standard_normal(num_frames),
                "nuisance_b": rng.standard_normal(num_frames),
                "extra_unused": np.arange(num_frames, dtype=float),
            },
        )
        confounds.loc[10, "task"] = 1.0
        confounds.loc[40, "task"] = 1.0

        model = Gaussian2DPRFModel(
            impulse_model=impulse_model,
            regressors_model=[
                ConvolvedRegressors(names=["task"], impulse_model=impulse_model),
                AdditiveRegressors(names=["nuisance_a", "nuisance_b"]),
            ],
        )

        for name in ("beta_task", "beta_nuisance_a", "beta_nuisance_b"):
            assert name in model.parameter_names

        impulse_param_counts = {p: model.parameter_names.count(p) for p in impulse_model.parameter_names}
        assert all(count == 1 for count in impulse_param_counts.values())

        params = base_params.copy()
        params["beta_task"] = [1.0, -1.0]
        params["beta_nuisance_a"] = [0.5, 0.5]
        params["beta_nuisance_b"] = [-1.0, 1.5]

        resp = np.asarray(model(stimulus, params, regressors=confounds))
        assert resp.shape == (self.num_units, num_frames)
        assert np.isfinite(resp).all()

    def test_regressors_required_when_model_set(
        self,
        stimulus: PRFStimulus,
        base_params: pd.DataFrame,
        impulse_model: DerivativeTwoGammaImpulse,
    ):
        """Calling without regressors when regressors_model is set raises ValueError."""
        model = Gaussian2DPRFModel(
            impulse_model=impulse_model,
            regressors_model=AdditiveRegressors(names=["mx"]),
        )
        params = base_params.copy()
        params["beta_mx"] = [0.0, 0.0]
        with pytest.raises(ValueError, match="regressors_model"):
            model(stimulus, params)

    def test_regressors_rejected_when_model_unset(
        self,
        stimulus: PRFStimulus,
        base_params: pd.DataFrame,
        impulse_model: DerivativeTwoGammaImpulse,
    ):
        """Passing regressors when regressors_model is None raises ValueError."""
        model = Gaussian2DPRFModel(impulse_model=impulse_model)
        num_frames = stimulus.design.shape[0]
        with pytest.raises(ValueError, match="regressors_model"):
            model(
                stimulus,
                base_params,
                regressors=pd.DataFrame({"mx": np.zeros(num_frames)}),
            )


class TestDoGIntegration:
    """Integration tests for regressors with the center-surround model."""

    num_units = 2

    @pytest.fixture
    def stimulus(self):
        """2D bar pRF stimulus."""
        return load_2d_prf_bar_stimulus()

    def test_additive_regressor_added(self, stimulus: PRFStimulus):
        """Adding an additive regressor shifts the DoG prediction by the regressor contribution."""
        num_frames = stimulus.design.shape[0]
        rng = np.random.default_rng(7)
        design = pd.DataFrame({"nuisance": rng.standard_normal(num_frames)})
        regressors = AdditiveRegressors(names=["nuisance"])

        baseline_model = DoG2DPRFModel()
        model = DoG2DPRFModel(regressors_model=regressors)

        base_params = pd.DataFrame(
            {
                "mu_x": [0.0, 1.0],
                "mu_y": [1.0, 0.0],
                "sigma_center": [1.0, 1.5],
                "sigma_surround": [5.0, 7.5],
                "delay": [6.0, 6.0],
                "dispersion": [0.9, 0.9],
                "undershoot": [12.0, 12.0],
                "u_dispersion": [0.9, 0.9],
                "ratio": [0.48, 0.48],
                "weight_deriv": [0.5, 0.5],
                "amplitude_center": [2.0, 1.2],
                "amplitude_surround": [-0.5, -0.3],
                "baseline": [0.1, -0.1],
            },
        )
        params = base_params.copy()
        params["beta_nuisance"] = [1.0, -1.0]

        baseline_resp = np.asarray(baseline_model(stimulus, base_params))
        full_resp = np.asarray(model(stimulus, params, regressors=design))

        betas = params[["beta_nuisance"]].to_numpy()
        expected_contribution = betas @ design[["nuisance"]].to_numpy().T

        assert np.allclose(full_resp - baseline_resp, expected_contribution, atol=1e-4)


class TestCFIntegration:
    """Integration tests for regressors with GaussianCFModel."""

    num_source = 6
    num_frames = 12

    @pytest.fixture
    def stimulus(self):
        """CF stimulus with a source response time series."""
        rng = np.random.default_rng(0)
        distances = np.abs(
            np.arange(self.num_source, dtype=float)[:, None] - np.arange(self.num_source, dtype=float)[None, :],
        )
        source_response = rng.standard_normal((self.num_source, self.num_frames))
        return CFStimulus(distance_matrix=distances, source_response=source_response)

    def test_additive_regressor_added(self, stimulus: CFStimulus):
        """An additive regressor adds to the CF prediction without affecting the baseline shape."""
        rng = np.random.default_rng(1)
        design = pd.DataFrame(
            rng.standard_normal((self.num_frames, 2)),
            columns=["nuisance_a", "nuisance_b"],
        )
        regressors = AdditiveRegressors(names=["nuisance_a", "nuisance_b"])

        baseline_model = GaussianCFModel()
        model = GaussianCFModel(regressors_model=regressors)

        base_params = pd.DataFrame(
            {
                "center_index": [0, 3],
                "sigma": [1.0, 1.5],
                "baseline": [0.1, -0.1],
                "amplitude": [1.0, 2.0],
            },
        )
        params = base_params.copy()
        params["beta_nuisance_a"] = [1.0, -0.5]
        params["beta_nuisance_b"] = [-0.5, 1.0]

        baseline_resp = np.asarray(baseline_model(stimulus, base_params))
        full_resp = np.asarray(model(stimulus, params, regressors=design))

        betas = params[["beta_nuisance_a", "beta_nuisance_b"]].to_numpy()
        expected_contribution = betas @ design[["nuisance_a", "nuisance_b"]].to_numpy().T

        assert full_resp.shape == baseline_resp.shape
        assert np.allclose(full_resp - baseline_resp, expected_contribution, atol=1e-4)
