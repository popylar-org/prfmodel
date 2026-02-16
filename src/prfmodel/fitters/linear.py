"""Linear fitters."""

import keras
import numpy as np
import pandas as pd
from keras import ops
from tqdm import tqdm
from prfmodel.models.base import BaseComposite
from prfmodel.stimuli.base import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype


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
        model: BaseComposite,
        stimulus: Stimulus,
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
        slope_name: str,
        intercept_name: str | None = None,
        batch_size: int | None = None,
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
        slope_name : str
            The name of the parameter that will be the estimated slope. Must refer to a column within `parameters`.
        intercept_name : str, optional
            The name of the parameter that will be the estimated intercept. Must refer to a column within `parameters`.
            If `None`, no intercept is estimated.
        batch_size : int, optional
            Number of data batches to fit at the same time. If `None` (the default), all batches are fit at once.

        Returns
        -------
        LeastSquaresHistory
            A history object that contains a dict with the key `loss` and the residual sum of squares
            for each data batch.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        if slope_name not in parameters.columns:
            msg = "Argument 'slope_name' must be a column in 'parameters'"
            raise ValueError(msg)

        if intercept_name is not None and intercept_name not in parameters.columns:
            msg = "Argument 'intercept_name' must be a column in 'parameters'"
            raise ValueError(msg)

        num_batches = len(parameters)
        if batch_size is None:
            batch_size = num_batches

        residual_sums = []
        new_parameters = parameters.copy()

        batch_starts = range(0, num_batches, batch_size)
        for start in tqdm(batch_starts, desc="Processing data batches", total=len(batch_starts)):
            end = min(start + batch_size, num_batches)
            batch_residuals, batch_params = self._fit_batch(
                data[start:end],
                new_parameters.iloc[start:end],
                slope_name,
                intercept_name,
            )
            new_parameters.iloc[start:end] = batch_params
            residual_sums.append(batch_residuals)

        return LeastSquaresHistory({"loss": np.concatenate(residual_sums)}), new_parameters

    def _fit_batch(
        self,
        data_batch: Tensor,
        parameter_batch: pd.DataFrame,
        slope_name: str,
        intercept_name: str | None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Fit a single data batch and return updated parameters."""
        data_batch = ops.convert_to_tensor(data_batch, dtype=self.dtype)

        parameter_batch = parameter_batch.copy()

        # Reset intercept and slope so that we can replace them with estimates
        # In order to estimate the correct intercepts and slopes, we first need to set them to zero and one,
        # respectively, so that model predictions are not modified by their current values; we replace them later
        # with the estimated values
        if intercept_name is not None:
            parameter_batch[intercept_name] = 0.0

        parameter_batch[slope_name] = 1.0

        predictions = self.model(self.stimulus, parameter_batch, self.dtype)

        if intercept_name is not None:
            intercept = ops.ones_like(predictions, dtype=self.dtype)
            x_list = [intercept, predictions]
        else:
            x_list = [predictions]

        x_matrix = ops.stack(x_list, axis=-1)

        targets = ops.expand_dims(data_batch, axis=-1)

        best_params = keras.ops.map(lambda x: keras.ops.lstsq(x[0], x[1]), (x_matrix, targets))

        residual_sum = ops.convert_to_numpy(ops.sum(ops.square(targets - x_matrix @ best_params), axis=(-2, -1)))

        best_params = ops.convert_to_numpy(best_params[..., 0])

        if intercept_name is not None:
            parameter_batch[intercept_name] = best_params[..., 0]
            parameter_batch[slope_name] = best_params[..., 1]
        else:
            parameter_batch[slope_name] = best_params[..., 0]

        return residual_sum, parameter_batch
