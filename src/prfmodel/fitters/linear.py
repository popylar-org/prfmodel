"""Linear fitters."""

import keras
import pandas as pd
from keras import ops
from prfmodel.models.base import BaseModel
from prfmodel.stimulus.prf import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype

_MAX_LINEAR_PARAMS = 2


class LeastSquaresHistory:
    """Least squares metric history.

    Logs losses and metrics over data batches resulting from least squares fitting.

    Attributes
    ----------
    history : dict
        Dictionary with keys indicating metric names and values containing metric values for each data batch.

    """

    def __init__(self, history: dict | None):
        self.history = history


class LeastSquaresFitter:
    """Fit population receptive field models with least squares.

    Estimates model parameters by minimizing the sum of least squares between model predictions and data.

    Parameters
    ----------
    model : BaseModel
        Population receptive field model instance that can be fit to data.
        The model must implement `__call__` to make predictions that can be compared to data.
    stimulus : Stimulus
        Stimulus object used to make model predictions.
    dtype : str, optional
        The dtype used for fitting. If `None` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.

    Notes
    -----
    This fitter only optmizes one or two model parameters by estimating a general linear model with an intercept and a
    slope between model predictions and data. Typically, these are baseline and amplitude parameters.

    Internally, the fitter applies `keras.ops.lstsq` to each data batch.

    """

    def __init__(
        self,
        model: BaseModel,
        stimulus: PRFStimulus,
        dtype: str | None = None,
    ):
        self.model = model
        self.stimulus = stimulus
        self.dtype = dtype

    @property
    def dtype(self) -> str:
        """The dtype that is used during fitting."""
        return self._dtype

    @dtype.setter
    def dtype(self, value: str | None) -> None:
        self._dtype = get_dtype(value)

    def fit(
        self,
        data: Tensor,
        parameters: pd.DataFrame,
        target_parameters: str | list[str],
    ) -> tuple[LeastSquaresHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : Tensor
            Target data to fit the model to. Must have shape (num_batches, num_frames), where `num_batches` is the
            number of batches for which parameters are estimated and `num_frames` is the number of time steps.
        parameters : pandas.DataFrame
            Dataframe with model parameters. Columns must contain different model parameters and
            rows parameter values for each batch in `data`.
        target_parameters : str or list of str
            The parameter(s) that will be optimized. Must either be a single parameter name or a list of one or
            two parameter names (intercept and slope).

        Returns
        -------
        LeastSquaresHistory
            A history object that contains a dict with the key `loss` and the residual sum of squares
            for each data batch.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        data = ops.convert_to_tensor(data, dtype=self.dtype)

        if not isinstance(target_parameters, (str, list)):
            msg = "Argument 'target_parameters' must either be a string or a list of strings"
            raise TypeError(msg)
        if isinstance(target_parameters, str):
            target_parameters = [target_parameters]

        predictions = self.model(self.stimulus, parameters)  # type: ignore[operator]

        if len(target_parameters) == _MAX_LINEAR_PARAMS:
            intercept = ops.ones_like(predictions)
            x_list = [intercept, predictions]
        elif len(target_parameters) == 1:
            x_list = [predictions]
        else:
            msg = "Length of 'target_parameters' argument must be 1 (slope-only) or 2 (intercept + slope)"
            raise ValueError(msg)

        x_matrix = ops.stack(x_list, axis=-1)

        targets = ops.expand_dims(data, axis=-1)

        best_params = keras.ops.map(lambda x: keras.ops.lstsq(x[0], x[1]), (x_matrix, targets))

        residual_sum = ops.convert_to_numpy(ops.sum(ops.square(targets - x_matrix @ best_params), axis=(-2, -1)))

        new_parameters = parameters.copy()

        new_parameters[target_parameters] = ops.convert_to_numpy(best_params[..., 0])

        return LeastSquaresHistory({"loss": residual_sum}), new_parameters
