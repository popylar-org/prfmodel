"""Linear fitters."""

import keras
import numpy as np
import pandas as pd
from keras import ops
from tqdm.auto import tqdm
from prfmodel._docstring import doc
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
    %(stimulus)s
    %(dtype)s

    Notes
    -----
    This fitter optimizes one or more slope parameters (and optionally an intercept) by estimating a general linear
    model between model predictions and data. Typically, these are baseline and amplitude parameters. When multiple
    slope names are given, each basis function is isolated by setting that slope to 1.0 and all others to 0.0, and
    the resulting design matrix is solved with least squares in one shot.

    Internally, the fitter applies `keras.ops.lstsq` to each data batch.

    Examples
    --------
    Fit a 2D Gaussian population receptive field model.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models.gaussian import Gaussian2DPRFModel
    >>> from prfmodel.fitters.linear import LeastSquaresFitter
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> print(stimulus)
    PRFStimulus(design=array[200, 101, 101], grid=array[101, 101, 2], dimension_labels=['y', 'x'])
    >>> # Only fit response and temporal model
    >>> model = Gaussian2DPRFModel(impulse_model=None)
    >>> # Define init parameters
    >>> params_init = pd.DataFrame({
    ...     "mu_x": [0.0], "mu_y": [0.0], "sigma": [1.0],
    ...     "baseline": [0.0], "amplitude": [0.0],
    ... })
    >>> # Create dummy data for a single unit
    >>> data = np.zeros((1, 200))
    >>> fitter = LeastSquaresFitter(model=model, stimulus=stimulus)
    >>> # Fit model parameters
    >>> history, params_ls = fitter.fit(
    ...     data,
    ...     params_init,
    ...     slope_name="amplitude",
    ...     intercept_name="baseline",
    ... )
    >>> print(list(params_ls.columns))
    ['mu_x', 'mu_y', 'sigma', 'baseline', 'amplitude']

    """

    @doc
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
        slope_name: str | list[str],
        intercept_name: str | None = None,
        batch_size: int | None = None,
    ) -> tuple[LeastSquaresHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : Tensor
            Target data to fit the model to. Must have shape (num_units, num_frames), where `num_units` is the
            number of units for which parameters are estimated and `num_frames` is the number of time steps.
        parameters : pandas.DataFrame
            Dataframe with model parameters. Columns must contain different model parameters and
            rows parameter values for each unit in `data`.
        slope_name : str or list of str
            The name(s) of the parameter(s) that will be the estimated slope(s). Must refer to column(s) within
            `parameters`.
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
        slope_names = [slope_name] if not isinstance(slope_name, list) else slope_name

        for name in slope_names:
            if name not in parameters.columns:
                msg = f"Slope name '{name}' must be a column in 'parameters'"
                raise ValueError(msg)

        if intercept_name is not None and intercept_name not in parameters.columns:
            msg = "Argument 'intercept_name' must be a column in 'parameters'"
            raise ValueError(msg)

        num_units = len(parameters)
        if batch_size is None:
            batch_size = num_units

        residual_sums = []
        new_parameters = parameters.copy()

        batch_starts = range(0, num_units, batch_size)
        for start in tqdm(batch_starts, desc="Processing data batches", total=len(batch_starts)):
            end = min(start + batch_size, num_units)
            batch_residuals, batch_params = self._fit_batch(
                data[start:end],
                new_parameters.iloc[start:end],
                slope_names,
                intercept_name,
            )
            new_parameters.iloc[start:end] = batch_params
            residual_sums.append(batch_residuals)

        return LeastSquaresHistory({"loss": np.concatenate(residual_sums)}), new_parameters

    def _fit_batch(
        self,
        data_batch: Tensor,
        parameter_batch: pd.DataFrame,
        slope_names: list[str],
        intercept_name: str | None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Fit a single data batch and return updated parameters."""
        data_batch = ops.convert_to_tensor(data_batch, dtype=self.dtype)

        parameter_batch = parameter_batch.copy()

        # Reset intercept and all slopes so that we can replace them with estimates
        if intercept_name is not None:
            parameter_batch[intercept_name] = 0.0

        for name in slope_names:
            parameter_batch[name] = 0.0

        # Build design matrix by isolating each basis function
        x_list = []

        for name in slope_names:
            parameter_batch[name] = 1.0
            predictions = self.model(self.stimulus, parameter_batch, self.dtype)
            x_list.append(predictions)
            parameter_batch[name] = 0.0

        if intercept_name is not None:
            x_list.insert(0, ops.ones_like(x_list[0], dtype=self.dtype))

        x_matrix = ops.stack(x_list, axis=-1)

        targets = ops.expand_dims(data_batch, axis=-1)

        best_params = keras.ops.map(lambda x: keras.ops.lstsq(x[0], x[1]), (x_matrix, targets))

        residual_sum = ops.convert_to_numpy(ops.sum(ops.square(targets - x_matrix @ best_params), axis=(-2, -1)))

        best_params = ops.convert_to_numpy(best_params[..., 0])

        # Assign coefficients back to parameters
        col_idx = 0
        if intercept_name is not None:
            parameter_batch[intercept_name] = best_params[..., col_idx]
            col_idx += 1

        for name in slope_names:
            parameter_batch[name] = best_params[..., col_idx]
            col_idx += 1

        return residual_sum, parameter_batch
