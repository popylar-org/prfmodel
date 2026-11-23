"""Custom loss functions.

Contains additional loss functions that inherit from `keras.losses.Loss`. They can be used inside
:class:`~prfmodel.fitters.GridFitter` and :class:`~prfmodel.fitters.SGDFitter`.

"""

from keras import ops
from keras.losses import Loss
from keras.losses import cosine_similarity
from prfmodel.exceptions import ShapeError
from prfmodel.typing import Tensor

# Number of dimensions of both inputs when `CorrelationLoss` compares every prediction with every target.
_PAIRWISE_NDIM = 3


class CorrelationLoss(Loss):
    """Correlation loss.

    Computes the negative Pearson correlation coefficient loss. This loss is agnostic to differences in baseline and
    amplitude between true values and predictions. This makes it particularly useful for
    :class:`~prfmodel.fitters.GridFitter` where it is the default loss.

    Parameters
    ----------
    pairwise : bool, default=False
        Whether to compare every prediction with every target instead of matching them element by element. With
        `True`, `y_true` must have shape `(1, num_targets, num_frames)` and `y_pred` shape
        `(num_predictions, 1, num_frames)`, and the loss has shape `(num_predictions, num_targets)` before reduction.
        This gives the same values as `False` with the same inputs, but computes them as a matrix product instead
        of broadcasting, which avoids an intermediate array of shape `(num_predictions, num_targets, num_frames)`.
        :class:`~prfmodel.fitters.GridFitter` uses this by default.
    **kwargs
        Keyword arguments passed to :class:`keras.losses.Loss`.

    Notes
    -----
    Computes the loss by first subtracting the mean from true values and predictions and then computing their
    cosine similarity. Different from the usual definition of the Pearson correlation coefficient, the loss is zero
    when either ``y_true`` or ``y_pred`` are constant (instead of ``NaN``).

    Because the loss is invariant to baseline and amplitude, parameters that only shift or scale the prediction
    are not identifiable under it. See the notes of :class:`~prfmodel.fitters.GridFitter` for the consequences.

    """

    def __init__(self, pairwise: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pairwise = pairwise

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Compute the loss.

        Parameters
        ----------
        y_true : :data:`prfmodel.typing.Tensor`
            Tensor of true targets.
        y_pred : :data:`prfmodel.typing.Tensor`
            Tensor of predicted targets.

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Correlation loss. Shape depends on the :attr:`reduction` argument in the class constructor.

        Raises
        ------
        ShapeError
            If `pairwise` is `True` and `y_true` or `y_pred` do not have the shapes it requires.

        """
        if self.pairwise:
            return _pairwise_correlation_loss(y_true, y_pred)

        return cosine_similarity(
            y_true - ops.mean(y_true, axis=-1, keepdims=True),
            y_pred - ops.mean(y_pred, axis=-1, keepdims=True),
        )

    def get_config(self) -> dict:
        """Return the configuration of the loss, including `pairwise`, for Keras serialization."""
        return {**super().get_config(), "pairwise": self.pairwise}


def _center_normalize(x: Tensor) -> Tensor:
    """Subtract the mean along the last axis, then scale to unit L2 norm along it.

    Uses :func:`keras.ops.normalize`, which is also what :func:`keras.losses.cosine_similarity` uses, so constant
    inputs become all zeros in the same way and give a loss of zero.

    """
    return ops.normalize(x - ops.mean(x, axis=-1, keepdims=True), axis=-1)


def _pairwise_correlation_loss(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """Compute the correlation loss between every prediction and every target as a matrix product.

    `y_true` has shape `(1, num_targets, num_frames)` and `y_pred` shape `(num_predictions, 1, num_frames)`, as
    produced by adding a size-one axis to each. The result has shape `(num_predictions, num_targets)`.

    """
    if len(y_true.shape) != _PAIRWISE_NDIM or y_true.shape[0] != 1:
        raise ShapeError("y_true", tuple(y_true.shape), "must have shape (1, num_targets, num_frames)")  # noqa: EM101 (exception literal)

    if len(y_pred.shape) != _PAIRWISE_NDIM or y_pred.shape[1] != 1:
        raise ShapeError("y_pred", tuple(y_pred.shape), "must have shape (num_predictions, 1, num_frames)")  # noqa: EM101 (exception literal)

    # Drop the size-one axes: targets (num_targets, num_frames), predictions (num_predictions, num_frames).
    y_true = _center_normalize(y_true[0])
    y_pred = _center_normalize(y_pred[:, 0])

    # Entry (i, j) is the dot product of normalized prediction i and normalized target j, which is their Pearson
    # correlation. Only the (num_predictions, num_targets) result is stored.
    return -ops.matmul(y_pred, ops.transpose(y_true))
