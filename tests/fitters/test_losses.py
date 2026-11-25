"""Tests for custom loss functions."""

import numpy as np
import pytest
from keras import ops
from prfmodel.exceptions import ShapeError
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


class TestPairwiseCorrelationLoss:
    """Tests for CorrelationLoss with `pairwise=True`.

    The inputs have the shapes :class:`~prfmodel.fitters.GridFitter` passes: targets `(1, num_targets, num_frames)`
    and predictions `(num_predictions, 1, num_frames)`. With these shapes, the loss with `pairwise=False` broadcasts
    to every pair as well, so it serves as the reference.

    """

    num_targets = 6
    num_predictions = 4
    num_frames = 30

    def _inputs(self, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y_true = rng.standard_normal((1, self.num_targets, self.num_frames))
        y_pred = rng.standard_normal((self.num_predictions, 1, self.num_frames))
        return y_true, y_pred

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_matches_broadcasting(self, dtype: str):
        """Test that the matrix product gives the same losses as broadcasting the element-wise loss."""
        y_true, y_pred = self._inputs()

        pairwise = CorrelationLoss(pairwise=True, reduction="none", dtype=dtype)(y_true, y_pred)
        broadcast = CorrelationLoss(reduction="none", dtype=dtype)(y_true, y_pred)

        assert pairwise.shape == (self.num_predictions, self.num_targets)
        np.testing.assert_allclose(ops.convert_to_numpy(pairwise), ops.convert_to_numpy(broadcast), atol=1e-6)

    def test_matches_pearson_correlation(self):
        """Test that each entry is the negative Pearson correlation of one prediction and one target."""
        y_true, y_pred = self._inputs()

        losses = ops.convert_to_numpy(CorrelationLoss(pairwise=True, reduction="none", dtype="float64")(y_true, y_pred))

        expected = -np.corrcoef(y_pred[:, 0], y_true[0])[: self.num_predictions, self.num_predictions :]
        np.testing.assert_allclose(losses, expected, atol=1e-6)

    def test_constant_time_courses_have_zero_loss(self):
        """Test that a constant target or a constant prediction gives a loss of zero, as without `pairwise`."""
        y_true, y_pred = self._inputs()
        y_true[0, 0] = 3.0
        y_pred[0, 0] = -1.0

        losses = ops.convert_to_numpy(CorrelationLoss(pairwise=True, reduction="none")(y_true, y_pred))

        # Zero only up to float32 precision: on some backends (e.g. JAX) the mean of a constant float32 series is
        # off by one ulp, and normalizing the tiny residual left after centering gives products around 1e-8.
        np.testing.assert_allclose(losses[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(losses[0, :], 0.0, atol=1e-6)

    def test_reduction(self):
        """Test that the default reduction averages over all pairs."""
        y_true, y_pred = self._inputs()

        reduced = CorrelationLoss(pairwise=True)(y_true, y_pred)
        unreduced = CorrelationLoss(pairwise=True, reduction="none")(y_true, y_pred)

        assert reduced.shape == ()
        np.testing.assert_allclose(ops.convert_to_numpy(reduced), ops.convert_to_numpy(unreduced).mean(), atol=1e-6)

    @pytest.mark.parametrize(
        ("true_shape", "pred_shape"),
        [
            # Matching shapes, as used without 'pairwise'
            ((num_targets, num_frames), (num_targets, num_frames)),
            # Targets not on the second axis
            ((2, num_targets, num_frames), (num_predictions, 1, num_frames)),
            # Predictions not on the first axis
            ((1, num_targets, num_frames), (num_predictions, 2, num_frames)),
        ],
    )
    def test_wrong_shapes_raise(self, true_shape: tuple[int, ...], pred_shape: tuple[int, ...]):
        """Test that inputs without the size-one axes `pairwise` requires raise instead of computing something else."""
        loss = CorrelationLoss(pairwise=True, reduction="none")

        with pytest.raises(ShapeError):
            loss(np.ones(true_shape), np.ones(pred_shape))

    @pytest.mark.parametrize("pairwise", [True, False])
    def test_config_round_trip(self, pairwise: bool):
        """Test that `pairwise` survives Keras serialization."""
        loss = CorrelationLoss(pairwise=pairwise, reduction="none")

        restored = CorrelationLoss.from_config(loss.get_config())

        assert restored.pairwise is pairwise
        assert restored.reduction == "none"
