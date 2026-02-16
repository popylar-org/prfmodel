"""Grid fitters."""

import math
import warnings
from collections.abc import Callable
from itertools import product
import keras
import numpy as np
import pandas as pd
from keras import ops
from more_itertools import chunked
from tqdm import tqdm
from prfmodel.models.base import BaseComposite
from prfmodel.stimuli.base import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict
from prfmodel.utils import get_dtype


class InfiniteLossWarning(UserWarning):
    """Warning for when infinite loss values are returned."""


class GridHistory:
    """Grid search metric history.

    Logs losses and metrics over data batches resulting from a grid search.

    Attributes
    ----------
    history : dict
        Dictionary with keys indicating metric names and values containing metric values for each data batch.

    """

    def __init__(self, history: dict | None):
        self.history = history


class GridFitter:
    """Fit population receptive field models with grid search.

    Estimates model parameters by evaluating the model on a grid of parameter combinations and finding the
    minimum loss.

    Parameters
    ----------
    model : BaseModel
        Population receptive field model instance that can be fit to data.
        The model must implement `__call__` to make predictions that can be compared to data.
    stimulus : Stimulus
        Stimulus object used to make model predictions.
    loss : keras.optimizers.Loss or Callable, optional
        Loss instance or function with the signature `f(y, y_pred)`, where `y` is the target data and `y_pred` are the
        model predicitons. Default is `None` where a `keras.optimizers.MeanSquaredError` loss is used. Note that, when
        a `keras.losses.Loss` instance is used, the argument `reduction` must be set to `"none"` to enable loss
        computation for all data batches.
    dtype : str, optional
        The dtype used for fitting. If `None` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.

    Notes
    -----
    Depending on the size of the parameter grid and the number of batches in the data, the search can be very
    memory-intensive. For this reason, the grid is first split into batches that are evaluated iteratively.

    """

    def __init__(
        self,
        model: BaseComposite,
        stimulus: Stimulus,
        loss: keras.losses.Loss | Callable | None = None,
        dtype: str | None = None,
    ):
        self.model = model
        self.stimulus = stimulus

        if loss is None:
            loss = keras.losses.MeanSquaredError(reduction="none")

        self.loss = loss
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
        parameter_values: dict[str, Tensor | np.ndarray],
        batch_size: int | None = None,
    ) -> tuple[GridHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : Tensor
            Target data to fit the model to. Must have shape (num_batches, num_frames), where `num_batches` is the
            number of batches for which parameters are estimated and `num_frames` is the number of time steps.
        parameter_values : dict
            Dictionary with keys indicating model parameters and values indicating parameter values in the grid. The
            grid is constructed by taking all combinations of parameters values (i.e., the cartesian product).
        batch_size : int, optional
            Number of parameter combinations to evaluate at the same time. If `None` (the default), all combinations
            are evaluated at once.

        Returns
        -------
        GridHistory
            A history object that contains loss and metric values for each data batch.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        data = ops.convert_to_tensor(data, dtype=self.dtype)

        parameter_names = list(parameter_values.keys())
        arrays = [ops.convert_to_numpy(val) for val in parameter_values.values()]
        total_grid_size = int(np.prod([len(x) for x in arrays]))

        if batch_size is None:
            batch_size = total_grid_size

        best_params = np.full((len(parameter_names), data.shape[0]), fill_value=np.nan, dtype=self.dtype)
        best_loss = np.full((data.shape[0],), fill_value=np.inf, dtype=self.dtype)
        data = ops.expand_dims(data, 0)

        param_iter = chunked(product(*arrays), n=batch_size)
        num_batches = math.ceil(total_grid_size / batch_size)

        with tqdm(param_iter, desc="Processing parameter grid", total=num_batches) as pbar:
            for batch in pbar:
                self._evaluate_parameter_batch(batch, parameter_names, data, best_params, best_loss)
                pbar.set_postfix({"loss": float(best_loss.mean())})

        if not all(ops.isfinite(best_loss)):
            msg = "Non-finite loss values and NaN parameter estimates returned for some data batches"
            warnings.warn(msg, category=InfiniteLossWarning)

        best_params_df = pd.DataFrame(best_params.T, columns=parameter_names)
        return GridHistory({"loss": best_loss}), best_params_df

    def _evaluate_parameter_batch(
        self,
        batch: list[tuple],
        parameter_names: list[str],
        data: Tensor,
        best_params: np.ndarray,
        best_loss: np.ndarray,
    ) -> None:
        """Evaluate a batch of parameter combinations and update best parameters if improved."""
        params = np.stack(batch).T
        param_dict = ParamsDict(dict(zip(parameter_names, params, strict=True)))

        # We ignore the arg type here because a ParamsDict is used internally (instead of pandas.DataFrame)
        pred = ops.expand_dims(self.model(self.stimulus, param_dict), 1)  # type: ignore[arg-type]
        losses = self.loss(data, pred)

        min_loss = ops.convert_to_numpy(ops.amin(losses, axis=0))
        min_loss_idx = ops.convert_to_numpy(ops.argmin(losses, axis=0))
        is_better = min_loss < best_loss
        best_loss[is_better] = min_loss[is_better]
        best_params[:, is_better] = params[:, min_loss_idx[is_better]]
