"""Tests for AdditiveRegressors."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.regressors import AdditiveRegressors
from tests.models.conftest import parametrize_dtype


class TestAdditiveRegressors:
    """Tests for AdditiveRegressors class."""

    num_frames = 10
    num_regressors = 3

    @pytest.fixture
    def names(self):
        """Regressor names."""
        return ["a", "b", "c"]

    @pytest.fixture
    def design(self, names: list[str]):
        """Random design DataFrame keyed by regressor name."""
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.standard_normal((self.num_frames, self.num_regressors)),
            columns=names,
        )

    @pytest.fixture
    def params(self):
        """Per-unit beta weights for two units."""
        return pd.DataFrame(
            {
                "beta_a": [1.0, -2.0],
                "beta_b": [0.5, 0.5],
                "beta_c": [-1.0, 1.5],
            },
        )

    def test_parameter_names(self, names: list[str]):
        """Test that parameter names are derived from regressor names."""
        regressors = AdditiveRegressors(names=names)
        assert regressors.parameter_names == ["beta_a", "beta_b", "beta_c"]

    @parametrize_dtype
    def test_call_shape_and_values(
        self,
        design: pd.DataFrame,
        names: list[str],
        params: pd.DataFrame,
        dtype: str,
    ):
        """Test that the output is `betas @ design.T` with the correct shape."""
        regressors = AdditiveRegressors(names=names)
        resp = np.asarray(regressors(design, params, dtype))

        assert resp.shape == (params.shape[0], self.num_frames)

        betas = params[regressors.parameter_names].to_numpy()
        expected = betas @ design[names].to_numpy().T
        assert np.allclose(resp, expected, atol=1e-5)

    def test_column_order_does_not_matter(
        self,
        design: pd.DataFrame,
        names: list[str],
        params: pd.DataFrame,
    ):
        """Test that shuffling design column order does not change the result."""
        regressors = AdditiveRegressors(names=names)
        resp_orig = np.asarray(regressors(design, params))

        shuffled = design[list(reversed(names))]
        resp_shuffled = np.asarray(regressors(shuffled, params))

        assert np.allclose(resp_orig, resp_shuffled, atol=1e-5)

    def test_extra_columns_ignored(
        self,
        design: pd.DataFrame,
        names: list[str],
        params: pd.DataFrame,
    ):
        """Test that designs containing extra columns are accepted and the extras are ignored."""
        regressors = AdditiveRegressors(names=names)
        augmented = design.copy()
        augmented["irrelevant"] = np.arange(self.num_frames, dtype=float)

        resp_base = np.asarray(regressors(design, params))
        resp_aug = np.asarray(regressors(augmented, params))

        assert np.allclose(resp_base, resp_aug, atol=1e-5)

    def test_missing_column_raises(self, names: list[str], params: pd.DataFrame):
        """Test that missing required columns raises ValueError listing the missing names."""
        regressors = AdditiveRegressors(names=names)
        partial = pd.DataFrame({"a": np.ones(self.num_frames), "b": np.ones(self.num_frames)})

        with pytest.raises(ValueError, match=r"missing required column"):
            regressors(partial, params)
