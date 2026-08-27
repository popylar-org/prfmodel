"""Least-squares fitters."""

from typing import cast
import numpy as np
import pandas as pd
from keras import ops
from tqdm.auto import tqdm
from prfmodel._docstring import doc
from prfmodel.models.base import BaseCanonical
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _extract_regressor_design
from prfmodel.regressors.base import _validate_regressors_argument
from prfmodel.stimuli import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import as_tensor_frame
from prfmodel.utils import get_dtype


def _eps(dtype: str) -> float:
    """Machine epsilon of a floating point dtype, used as the relative cutoff on singular values."""
    return float(np.finfo(dtype).eps)


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


@doc
class LeastSquaresFitter:
    """Fit population receptive field models with least squares.

    Estimates model parameters by minimizing the sum of least squares between model predictions and data.

    Parameters
    ----------
    %(model_fitter)s
    %(stimulus)s
    %(dtype)s

    Notes
    -----
    This fitter optimizes one or more slope parameters (and optionally an intercept) by estimating a general linear
    model between model predictions and data. Typically, these are baseline and amplitude parameters. When multiple
    slope names are given, each basis function is isolated by setting that slope to 1.0 and all others to 0.0, and
    the resulting design matrix is solved with least squares in one shot.

    Internally, each data batch is solved for all of its units at once with a batched singular value
    decomposition, which gives the same coefficients as :func:`numpy.linalg.lstsq` per unit, including the
    minimum-norm solution when a unit's design matrix is rank deficient.

    Examples
    --------
    Fit a 2D Gaussian population receptive field model.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models.prf import Gaussian2DPRFModel
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> # Only fit response and temporal model
    >>> model = Gaussian2DPRFModel(impulse_model=None)
    >>> # Define init parameters
    >>> params_init = pd.DataFrame({
    ...     "mu_x": [0.0], "mu_y": [0.0], "sigma": [1.0],
    ...     "baseline": [0.0], "amplitude": [0.0],
    ... })
    >>> # Create dummy data for a single unit
    >>> data = np.zeros((1, stimulus.design.shape[0]))
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
        model: BaseCanonical,
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

    def fit(  # noqa: PLR0913 (too many arguments)
        self,
        data: Tensor,
        parameters: pd.DataFrame,
        slope_name: str | list[str],
        intercept_name: str | None = None,
        batch_size: int | None = None,
        regressors: pd.DataFrame | None = None,
    ) -> tuple[LeastSquaresHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : :data:`prfmodel.typing.Tensor`
            Target data to fit the model to. Must have shape `(num_units, num_frames)`, where `num_units` is the
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
        regressors : pandas.DataFrame, optional
            Regressor design data. Required when the model has a ``regressors_model`` configured.
            A single data frame with shape `(num_frames, num_regressors)` whose columns cover the names required
            by every configured regressor model; extra columns are ignored.

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

        regressors_model = cast("BaseRegressors | None", self.model.models.get("regressors_model"))

        _validate_regressors_argument(regressors_model, regressors)

        if regressors is not None and regressors_model is not None:
            # Beta weights must be in slope names if regressors are present, otherwise estimates are biased.
            # Only the columns the regressors model actually reads imply a beta weight; a column the caller
            # carried along in the same frame has none and must not be demanded as a slope.
            regressor_names = regressors_model.regressor_names
            name_diff = {f"beta_{name}" for name in regressor_names}.difference(set(slope_names))
            if len(name_diff) > 0:
                msg = f"No beta weight parameters found for regressors: {list(name_diff)}"
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
                regressors,
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
        regressors: pd.DataFrame | None,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Fit a single data batch and return updated parameters."""
        data_batch = ops.convert_to_tensor(data_batch, dtype=self.dtype)

        parameter_batch = parameter_batch.copy()

        # Reset intercept and all slopes so that we can replace them with estimates
        if intercept_name is not None:
            parameter_batch[intercept_name] = 0.0

        for name in slope_names:
            parameter_batch[name] = 0.0

        # Build design matrix by isolating each basis function.
        # The stimulus is converted once and the model is entered through 'call', because this loop runs the
        # same model once per slope and the facade would re-validate and re-convert the stimulus every time.
        stimulus = self.stimulus.to_tensors(self.dtype)
        regressors_model = cast("BaseRegressors | None", self.model.models.get("regressors_model"))
        regressors_consumed = _extract_regressor_design(regressors_model, regressors, self.dtype)

        x_list = []

        for name in slope_names:
            parameter_batch[name] = 1.0
            # Validate parameters outside of the models call method
            self.model.check_parameter_names(parameter_batch)
            self.model.check_parameter_values(parameter_batch)
            params = as_tensor_frame(
                parameter_batch[self.model.get_consumed_parameter_names(parameter_batch)],
                self.dtype,
            )
            x_list.append(self.model.call(stimulus, params, regressors=regressors_consumed))
            parameter_batch[name] = 0.0

        if intercept_name is not None:
            x_list.insert(0, ops.ones_like(x_list[0], dtype=self.dtype))

        x_matrix = ops.stack(x_list, axis=-1)

        targets = ops.expand_dims(data_batch, axis=-1)

        # One least-squares problem per unit, solved for all units at once with a batched singular value
        # decomposition.
        u_factor, singular_values, vt_factor = ops.linalg.svd(x_matrix, full_matrices=False)

        # Singular values at or below the cutoff count as zero, which is what 'numpy.linalg.lstsq' does
        # with 'rcond=None'. A unit whose design matrix is rank deficient (i.e., a prediction that is constant,
        # alongside an estimated intercept) then gets the minimum-norm solution rather than an infinity.
        # The reciprocal is guarded so that the discarded branch never divides by zero.
        cutoff = ops.amax(singular_values, axis=-1, keepdims=True) * (max(x_matrix.shape[-2:]) * _eps(self.dtype))
        is_nonzero = singular_values > cutoff
        inverse_values = ops.where(is_nonzero, 1.0 / ops.where(is_nonzero, singular_values, 1.0), 0.0)

        projected = ops.expand_dims(inverse_values, axis=-1) * (ops.transpose(u_factor, (0, 2, 1)) @ targets)
        best_params = ops.transpose(vt_factor, (0, 2, 1)) @ projected

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
