"""Tests for ConvolvedRegressors."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.regressors import ConvolvedRegressors
from tests.models.conftest import parametrize_dtype


class TestConvolvedRegressors:
    """Tests for ConvolvedRegressors class."""

    num_frames = 20
    num_regressors = 2
    num_units = 2

    @pytest.fixture
    def names(self):
        """Regressor names."""
        return ["task_a", "task_b"]

    @pytest.fixture
    def design(self, names: list[str]):
        """Design DataFrame with two distinct event time courses."""
        arr = np.zeros((self.num_frames, self.num_regressors))
        arr[3, 0] = 1.0
        arr[7, 0] = 1.0
        arr[5, 1] = 1.0
        arr[12, 1] = 1.0
        return pd.DataFrame(arr, columns=names)

    @pytest.fixture
    def impulse_model(self):
        """Impulse model instance."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def params(self):
        """Parameters dataframe with betas and impulse parameters."""
        impulse_params = {
            "delay": [6.0, 6.0],
            "dispersion": [0.9, 0.9],
            "undershoot": [12.0, 12.0],
            "u_dispersion": [0.9, 0.9],
            "ratio": [0.48, 0.48],
            "weight_deriv": [0.5, 0.5],
        }

        return pd.DataFrame(
            {
                "beta_task_a": [1.0, -1.0],
                "beta_task_b": [0.5, 2.0],
                **impulse_params,
            },
        )

    def test_parameter_names_includes_impulse_params(
        self,
        names: list[str],
        impulse_model: DerivativeTwoGammaImpulse,
    ):
        """Test that parameter names contain beta_<name> and the impulse model's parameters."""
        regressors = ConvolvedRegressors(names=names, impulse_model=impulse_model)
        assert "beta_task_a" in regressors.parameter_names
        assert "beta_task_b" in regressors.parameter_names
        for name in impulse_model.parameter_names:
            assert name in regressors.parameter_names

    @parametrize_dtype
    def test_call_shape_and_values(
        self,
        design: pd.DataFrame,
        names: list[str],
        impulse_model: DerivativeTwoGammaImpulse,
        params: pd.DataFrame,
        dtype: str,
    ):
        """Test that output equals the manual sum of beta_k * (regressor_k * HRF)(t)."""
        regressors = ConvolvedRegressors(names=names, impulse_model=impulse_model)
        resp = np.asarray(regressors(design, params, dtype))

        assert resp.shape == (self.num_units, self.num_frames)

        impulse = np.asarray(impulse_model(params, dtype=dtype))
        design_np = design[names].to_numpy()
        expected = np.zeros((self.num_units, self.num_frames))
        for idx in range(self.num_regressors):
            reg = np.broadcast_to(design_np[:, idx], (self.num_units, self.num_frames))
            convolved = np.asarray(convolve_prf_impulse_response(reg, impulse, dtype=dtype))
            beta = params[f"beta_{names[idx]}"].to_numpy()[:, None]
            expected = expected + beta * convolved

        assert np.allclose(resp, expected, atol=1e-4)

    def test_column_order_does_not_matter(
        self,
        design: pd.DataFrame,
        names: list[str],
        impulse_model: DerivativeTwoGammaImpulse,
        params: pd.DataFrame,
    ):
        """Test that shuffling design column order does not change the result."""
        regressors = ConvolvedRegressors(names=names, impulse_model=impulse_model)
        resp_orig = np.asarray(regressors(design, params))

        shuffled = design[list(reversed(names))]
        resp_shuffled = np.asarray(regressors(shuffled, params))

        assert np.allclose(resp_orig, resp_shuffled, atol=1e-5)

    def test_extra_columns_ignored(
        self,
        design: pd.DataFrame,
        names: list[str],
        impulse_model: DerivativeTwoGammaImpulse,
        params: pd.DataFrame,
    ):
        """Test that designs containing extra columns are accepted and the extras are ignored."""
        regressors = ConvolvedRegressors(names=names, impulse_model=impulse_model)
        augmented = design.copy()
        augmented["irrelevant"] = np.arange(self.num_frames, dtype=float)

        resp_base = np.asarray(regressors(design, params))
        resp_aug = np.asarray(regressors(augmented, params))

        assert np.allclose(resp_base, resp_aug, atol=1e-5)

    def test_missing_column_raises(
        self,
        names: list[str],
        impulse_model: DerivativeTwoGammaImpulse,
        params: pd.DataFrame,
    ):
        """Test that missing required columns raises ValueError listing the missing names."""
        regressors = ConvolvedRegressors(names=names, impulse_model=impulse_model)
        partial = pd.DataFrame({"task_a": np.zeros(self.num_frames)})

        with pytest.raises(ValueError, match=r"missing required column"):
            regressors(partial, params)
