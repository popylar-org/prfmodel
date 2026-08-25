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
from tqdm.auto import tqdm
from prfmodel._backend import compile_fun
from prfmodel._docstring import doc
from prfmodel.fitters.losses import CorrelationLoss
from prfmodel.models.base import BaseCanonical
from prfmodel.regressors.base import _validate_regressors_argument
from prfmodel.stimuli import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import as_tensor_frame
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
    %(model_fitter)s
    %(stimulus)s
    loss : keras.optimizers.Loss or Callable, optional
        Loss instance or function with the signature `f(y_true, y_pred)`, where `y_true` is the target data and
        `y_pred` are the model predicitons. Default is `None` where a
        :class:`~prfmodel.fitters.losses.CorrelationLoss` loss is used. Note that, when a
        :class:`keras.losses.Loss` instance is used, the argument `reduction` must be set to `"none"` to enable
        loss computation for all data batches. When a function is used, it must return a loss value for every unit in
        `y_true` and `y_pred`.
    compile_step : bool, default=False
        Whether to compile the optimization step with the backend's native primitive: `jax.jit` on JAX,
                `tf.function` on TensorFlow and `torch.compile` on PyTorch. With `False` (the default) the step
                runs eagerly, which is slower but makes the step inspectable and produces Python
                tracebacks that point at the offending line. Whether compiling pays off depends on the backend
                and on the problem size.
    %(dtype)s

    Notes
    -----
    Depending on the size of the parameter grid and the number of batches in the data, the search can be very
    memory-intensive. For this reason, the grid is first split into batches that are evaluated iteratively.

    When `compile_step` is enabled, the batch evaluation is compiled once and reused for every batch. When the
    grid size is not a multiple of `batch_size` the final batch is smaller and is traced a second time.

    The default :class:`~prfmodel.fitters.losses.CorrelationLoss` is invariant to the baseline and amplitude of the
    prediction, so the target data does not need to be demeaned or converted to percent signal change beforehand.
    The flip side is that parameters which only shift or scale the prediction (typically `baseline` and `amplitude`)
    are not identifiable: all their grid values yield the same loss, and the returned estimate is an arbitrary one
    among them. Fix such parameters to a single placeholder value in the grid and estimate them afterwards with
    :class:`~prfmodel.fitters.LeastSquaresFitter`, or pass a loss that is sensitive to scale (e.g.,
    :class:`keras.losses.MeanSquaredError`) if they must be estimated by the grid search itself.

    Examples
    --------
    Fit a 2D Gaussian population receptive field model.

    >>> import numpy as np
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models.prf import Gaussian2DPRFModel
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> # Only fit response model
    >>> model = Gaussian2DPRFModel(
    ...     impulse_model=None,
    ...     scaling_model=None,
    ... )
    >>> fitter = GridFitter(model=model, stimulus=stimulus)
    >>> # Create dummy data for a single unit
    >>> data = np.zeros((1, stimulus.design.shape[0]))
    >>> # Define possible parameters in grid
    >>> param_values = {"mu_y": [0.0], "mu_x": [0.0], "sigma": [1.0]}
    >>> # Fit model parameters
    >>> history, params_grid = fitter.fit(data, param_values)
    >>> print(list(params_grid.columns))
    ['mu_y', 'mu_x', 'sigma']
    >>> print(params_grid.shape)
    (1, 3)

    """

    @doc
    def __init__(
        self,
        model: BaseCanonical,
        stimulus: Stimulus,
        loss: keras.losses.Loss | Callable | None = None,
        compile_step: bool = False,
        dtype: str | None = None,
    ):
        self.model = model
        self.stimulus = stimulus

        if loss is None:
            loss = CorrelationLoss(reduction="none")

        self.loss = loss
        self.compile_step = compile_step
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
        regressors: pd.DataFrame | None = None,
    ) -> tuple[GridHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : :data:`prfmodel.typing.Tensor`
            Target data to fit the model to. Must have shape `(num_units, num_frames)`, where `num_units` is the
            number of units for which parameters are estimated and `num_frames` is the number of time steps.
        parameter_values : dict
            Dictionary with keys indicating model parameters and values indicating parameter values in the grid. The
            grid is constructed by taking all combinations of parameters values (i.e., the cartesian product).
        batch_size : int, optional
            Number of parameter combinations to evaluate at the same time. If `None` (the default), all combinations
            are evaluated at once.
        regressors : pandas.DataFrame, optional
            Runtime regressor design data forwarded to every model prediction call. Must be provided when the model
            has a ``regressors_model`` configured.

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

        _validate_regressors_argument(self.model.models.get("regressors_model"), regressors)
        self._validate_grid(parameter_names, arrays)

        if batch_size is None:
            batch_size = total_grid_size

        best_params = np.full((len(parameter_names), data.shape[0]), fill_value=np.nan, dtype=self.dtype)
        best_loss = np.full((data.shape[0],), fill_value=np.inf, dtype=self.dtype)
        data = ops.expand_dims(data, 0)

        param_iter = chunked(product(*arrays), n=batch_size)
        num_batches = math.ceil(total_grid_size / batch_size)

        # Built once and reused for every batch, so the cost of tracing is paid once rather than
        # 'num_batches' times. The data and the regressors are the same on every batch, so they are
        # captured here instead of being passed per batch, which leaves the parameter tensors as the only
        # argument: a 'pandas.DataFrame' is not a valid argument to a compiled function.
        evaluate = self._make_evaluate_fun(data, regressors)

        with tqdm(param_iter, desc="Processing parameter grid", total=num_batches) as pbar:
            for batch in pbar:
                self._evaluate_parameter_batch(evaluate, batch, parameter_names, best_params, best_loss)
                pbar.set_postfix({"loss": float(best_loss.mean())})

        if not all(ops.isfinite(best_loss)):
            msg = "Non-finite loss values and NaN parameter estimates returned for some data batches"
            warnings.warn(msg, category=InfiniteLossWarning)

        best_params_df = pd.DataFrame(best_params.T, columns=parameter_names)
        return GridHistory({"loss": best_loss}), best_params_df

    def _validate_grid(
        self,
        parameter_names: list[str],
        arrays: list[np.ndarray],
    ) -> None:
        # The batches are evaluated through 'call', which must stay traceable and so validates nothing. These
        # are the checks the public model facade would have run, before the loop. The grid is a
        # cartesian product and every domain check is elementwise on a single parameter, so checking each axis
        # once covers every combination the loop will evaluate. Each axis is tiled to a common length rather
        # than expanded into the product, which keeps this proportional to the longest axis.
        num_rows = max((len(array) for array in arrays), default=0)
        grid = pd.DataFrame(
            {name: np.resize(array, num_rows) for name, array in zip(parameter_names, arrays, strict=True)},
        )

        self.model._check_parameters(grid)  # noqa: SLF001 (fitter stands in for the model facade)
        self.model._check_parameter_values(grid)  # noqa: SLF001 (fitter stands in for the model facade)

    def _make_evaluate_fun(self, data: Tensor, regressors: pd.DataFrame | None) -> Callable:
        stimulus = self.stimulus.to_tensors(self.dtype)
        regressor_params = None if regressors is None else as_tensor_frame(regressors, self.dtype)

        def evaluate(param_tensors: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
            params = TensorFrame(param_tensors, dtype=self.dtype)
            pred = ops.expand_dims(self.model.call(stimulus, params, regressors=regressor_params), 1)
            losses = self.loss(data, pred)

            return ops.amin(losses, axis=0), ops.argmin(losses, axis=0)

        return compile_fun(evaluate) if self.compile_step else evaluate

    def _evaluate_parameter_batch(  # noqa: PLR0913 (too many arguments)
        self,
        evaluate: Callable,
        batch: list[tuple],
        parameter_names: list[str],
        best_params: np.ndarray,
        best_loss: np.ndarray,
    ) -> None:
        """Evaluate a batch of parameter combinations and update best parameters if improved."""
        params = np.stack(batch).T
        param_tensors = {
            name: ops.convert_to_tensor(values, dtype=self.dtype)
            for name, values in zip(parameter_names, params, strict=True)
        }

        min_loss, min_loss_idx = evaluate(param_tensors)

        min_loss = ops.convert_to_numpy(min_loss)
        min_loss_idx = ops.convert_to_numpy(min_loss_idx)
        is_better = min_loss < best_loss
        best_loss[is_better] = min_loss[is_better]
        best_params[:, is_better] = params[:, min_loss_idx[is_better]]
