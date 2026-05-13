"""Tests for RegressorsList."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.regressors import AdditiveRegressors
from prfmodel.regressors import RegressorsList


class TestRegressorsList:
    """Tests for RegressorsList class."""

    num_frames = 8

    @pytest.fixture
    def regressor_a(self):
        """First child regressor model."""
        return AdditiveRegressors(names=["x"])

    @pytest.fixture
    def regressor_b(self):
        """Second child regressor model."""
        return AdditiveRegressors(names=["y", "z"])

    def test_parameter_names_dedup_and_order(
        self,
        regressor_a: AdditiveRegressors,
        regressor_b: AdditiveRegressors,
    ):
        """Parameter names are aggregated in order, with duplicates removed."""
        regressors = RegressorsList([regressor_a, regressor_b])
        assert regressors.parameter_names == ["beta_x", "beta_y", "beta_z"]

    def test_call(
        self,
        regressor_a: AdditiveRegressors,
        regressor_b: AdditiveRegressors,
    ):
        """One combined DataFrame is broadcast to every child, and contributions sum correctly."""
        regressors = RegressorsList([regressor_a, regressor_b])
        params = pd.DataFrame({"beta_x": [1.0], "beta_y": [1.0], "beta_z": [-1.0]})

        design = pd.DataFrame(
            {
                "x": np.ones(self.num_frames),
                "y": np.ones(self.num_frames) * 2.0,
                "z": np.ones(self.num_frames) * 3.0,
            },
        )

        resp = np.asarray(regressors(design, params))
        expected = np.asarray(regressor_a(design[["x"]], params)) + np.asarray(regressor_b(design[["y", "z"]], params))

        assert resp.shape == (1, self.num_frames)
        assert np.allclose(resp, expected)

    def test_empty_list_raises(self):
        """An empty regressors list raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            RegressorsList([])

    def test_invalid_element_raises(self, regressor_a: AdditiveRegressors):
        """A non-BaseRegressors element raises TypeError."""
        with pytest.raises(TypeError, match="BaseRegressors"):
            RegressorsList([regressor_a, "not a regressor"])  # type: ignore[list-item]
