"""Tests for custom loss functions."""

import numpy as np
import pytest
from prfmodel.fitters.losses import CorrelationLoss


class TestCorrelationLoss:
    """Tests for the CorrelationLoss class."""

    num_units = 10
    num_frames = 5

    @pytest.fixture
    def loss(self) -> CorrelationLoss:
        """Create a loss instance."""
        return CorrelationLoss()

    def test_perfect_correlation(self, loss: CorrelationLoss):
        """Test that two identical signals are perfectly positively correlated."""
        y_true = np.linspace(-5.0, 5.0, self.num_units)
        y_pred = np.copy(y_true)

        assert np.isclose(loss(y_true, y_pred), -1.0)  # loss is negative correlation

    def test_perfect_inverse_correlation(self, loss: CorrelationLoss):
        """Test that two flipped signals are perfectly negatively correlated."""
        y_true = np.linspace(-5.0, 5.0, self.num_units)
        y_pred = np.flip(np.copy(y_true))

        assert np.isclose(loss(y_true, y_pred), 1.0)

    def test_random_matches_pearson_correlation(self, loss: CorrelationLoss):
        """Test that the loss of two random signals is the negative of their Pearson correlation."""
        rng = np.random.default_rng(2026)
        y_true = rng.random(self.num_units)
        y_pred = rng.random(self.num_units)

        loss_val = loss(y_true, y_pred)

        assert np.isclose(loss_val, -np.corrcoef(y_true, y_pred)[0, 1], atol=1e-6)

    def test_zero_variance_no_correlation(self, loss: CorrelationLoss):
        """Test that two signals with zero variance are not correlated."""
        y_true = np.ones((self.num_units,))
        y_pred = np.copy(y_true)

        assert np.isclose(loss(y_true, y_pred), 0.0)

    def test_baseline_shift(self, loss: CorrelationLoss):
        """Test that two baseline-shifted signals are perfectly positively correlated."""
        y_true = np.linspace(-5.0, 5.0, self.num_units)
        y_pred = np.copy(y_true) + 10

        assert np.isclose(loss(y_true, y_pred), -1.0)

    def test_amplitude_scaling(self, loss: CorrelationLoss):
        """Test that two amplitude-scaled signals are perfectly positively correlated."""
        y_true = np.linspace(-5.0, 5.0, self.num_units)
        y_pred = np.copy(y_true) * 10

        assert np.isclose(loss(y_true, y_pred), -1.0)

    def test_batching_with_reduction(self, loss: CorrelationLoss):
        """Test that reduction returns a single loss value for a batch."""
        y_true = np.ones((self.num_units, self.num_frames))
        y_pred = np.copy(y_true)

        loss_val = loss(y_true, y_pred)

        assert loss_val.shape == ()

    def test_batching_without_reduction(self):
        """Test that no reduction returns a loss value for each unit in a batch."""
        loss = CorrelationLoss(reduction="none")
        y_true = np.ones((self.num_units, self.num_frames))
        y_pred = np.copy(y_true)

        loss_arr = loss(y_true, y_pred)

        assert loss_arr.shape == (self.num_units,)
