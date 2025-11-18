"""Grid fitters."""

from collections.abc import Callable
from collections.abc import Iterator
from itertools import product
from typing import NamedTuple
import keras
import numpy as np
import pandas as pd
from keras import ops
from more_itertools import chunked
from tqdm import tqdm
from prfmodel.models.base import BaseModel
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict
from prfmodel.utils import get_dtype


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


class _GridFitState(NamedTuple):
    """Internal state container for grid fitting process."""

    parameter_names: list[str]
    data: Tensor
    best_params: np.ndarray
    best_loss: np.ndarray
    total_grid_size: int
    chunk_size: int


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
    memory-intensive. For this reason, the grid is first split into chunks that are evaluated iteratively.

    """

    def __init__(
        self,
        model: BaseModel,
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
        chunk_size: int | None = None,
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
        chunk_size : int, optional
            Size of each chunk of the grid that is evaluated at the same time.

        Returns
        -------
        GridHistory
            A history object that contains loss and metric values for each data batch.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        # Prepare data and parameter grid
        data = ops.convert_to_tensor(data, dtype=self.dtype)
        arrays = [ops.convert_to_numpy(val) for val in parameter_values.values()]
        parameter_names = list(parameter_values.keys())
        total_grid_size, chunk_size = self._compute_grid_dimensions(arrays, chunk_size)
        param_iter_batched = self._create_parameter_iterator(arrays, chunk_size)
        best_params, best_loss = self._initialize_best_tracking(parameter_names, data)
        data = ops.expand_dims(data, 0)

        # Create state object and process parameter grid chunks
        state = _GridFitState(
            parameter_names=parameter_names,
            data=data,
            best_params=best_params,
            best_loss=best_loss,
            total_grid_size=total_grid_size,
            chunk_size=chunk_size,
        )
        self._process_parameter_chunks(param_iter_batched, state)

        # Format and return results
        best_params_df = pd.DataFrame(best_params.T, columns=parameter_names)
        return GridHistory({"loss": best_loss}), best_params_df

    def _compute_grid_dimensions(
        self,
        arrays: list[np.ndarray],
        chunk_size: int | None,
    ) -> tuple[int, int]:
        """Compute total grid size and determine chunk size."""
        total_grid_size = int(np.prod([len(x) for x in arrays]))
        if chunk_size is None:
            chunk_size = total_grid_size
        return total_grid_size, chunk_size

    def _create_parameter_iterator(
        self,
        arrays: list[np.ndarray],
        chunk_size: int,
    ) -> Iterator[list[tuple]]:
        """Create a batched iterator over parameter combinations."""
        return chunked(product(*arrays), n=chunk_size)

    def _initialize_best_tracking(
        self,
        parameter_names: list[str],
        data: Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize arrays to track best parameters and losses."""
        best_params = np.empty((len(parameter_names), data.shape[0]), dtype=self.dtype)
        best_loss = np.full((data.shape[0],), fill_value=np.inf, dtype=self.dtype)
        return best_params, best_loss

    def _process_parameter_chunks(self, param_iter_batched: Iterator[list[tuple]], state: _GridFitState) -> None:
        """Process parameter grid chunks and update best parameters and losses."""
        with tqdm(
            param_iter_batched,
            desc="Processing parameter grid chunks",
            total=int(state.total_grid_size / state.chunk_size),
        ) as pbar:
            for batch in pbar:
                self._evaluate_parameter_batch(batch, state)
                pbar.set_postfix({"loss": float(state.best_loss.mean())})

    def _evaluate_parameter_batch(self, batch: list[tuple], state: _GridFitState) -> None:
        """Evaluate a batch of parameter combinations and update best parameters if improved."""
        # Stack tuples from generator and create parameter dictionary
        params = np.stack(batch).T
        param_dict = ParamsDict(dict(zip(state.parameter_names, params, strict=True)))

        # Generate predictions and calculate losses
        pred = ops.expand_dims(self.model(self.stimulus, param_dict), 1)  # type: ignore[operator]
        losses = self.loss(state.data, pred)

        # Update best parameters and losses where improvement is found
        self._update_best_parameters(losses, params, state)

    def _update_best_parameters(self, losses: Tensor, params: np.ndarray, state: _GridFitState) -> None:
        """Update best parameters and losses based on new losses."""
        min_loss = ops.convert_to_numpy(ops.amin(losses, axis=0))
        min_loss_idx = ops.convert_to_numpy(ops.argmin(losses, axis=0))
        loss_is_lower = min_loss < state.best_loss
        state.best_loss[loss_is_lower] = min_loss[loss_is_lower]
        state.best_params[:, loss_is_lower] = params[:, min_loss_idx[loss_is_lower]]
