"""Custom loss functions.

Contains additional loss functions that inherit from `keras.losses.Loss`. They can be used inside
:class:`~prfmodel.fitters.GridFitter` and :class:`~prfmodel.fitters.SGDFitter`.

"""

from keras import ops
from keras.losses import Loss
from keras.losses import cosine_similarity
from prfmodel.typing import Tensor


class CorrelationLoss(Loss):
    """Correlation loss.

    Computes the negative Pearson correlation coefficient loss. This loss is agnostic to differences in baseline and
    amplitude between true values and predictions. This makes it particularly useful for
    :class:`~prfmodel.fitters.GridFitter` where it is the default loss.

    Inherits all arguments from :class:`keras.losses.Loss`.

    Notes
    -----
    Computes the loss by first subtracting the mean from true values and predictions and then computing their
    cosine similarity. Different from the usual definition of the Pearson correlation coefficient, the loss is zero
    when either ``y_true`` or ``y_pred`` are constant (instead of ``NaN``).

    Because the loss is invariant to baseline and amplitude, parameters that only shift or scale the prediction
    are not identifiable under it. See the notes of :class:`~prfmodel.fitters.GridFitter` for the consequences.

    """

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

        """
        return cosine_similarity(
            y_true - ops.mean(y_true, axis=-1, keepdims=True),
            y_pred - ops.mean(y_pred, axis=-1, keepdims=True),
        )
