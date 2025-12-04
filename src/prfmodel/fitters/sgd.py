"""Stochastic gradient descent fitters."""

from collections.abc import Callable
from collections.abc import Sequence
import keras
import pandas as pd
from keras import ops
from tqdm import tqdm
from prfmodel.adapter import Adapter
from prfmodel.models.base import BaseModel
from prfmodel.stimulus.prf import PRFStimulus
from prfmodel.typing import Tensor

match keras.backend.backend():
    case "jax":
        from .backend.jax import JAXSGDFitter as BackendSGDFitter
    case "tensorflow":
        from .backend.tensorflow import TensorFlowSGDFitter as BackendSGDFitter
    case "torch":
        from .backend.torch import TorchSGDFitter as BackendSGDFitter
    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)


class SGDHistory:
    """Stochastic gradient descent metric history.

    Logs losses and metrics over stochastic gradient descent steps.

    Attributes
    ----------
    history : dict
        Dictionary with keys indicating metric names and values containing metric values for each step.
    step : list of int
        List of step indices.

    """

    def __init__(self):
        self.history = {}
        self.step = []

    def on_step_end(self, step: int, logs: dict | None) -> None:
        """Append metrics to the history after a stochastic gradient descent step.

        Parameters
        ----------
        step : int
            Step index.
        logs : dict
            Dictionary with keys indicating metric names and values containing the metric value for the step.

        """
        # Adapted from keras.callbacks.History:
        # https://github.com/keras-team/keras/blob/master/keras/src/callbacks/history.py
        logs = logs or {}
        self.step.append(step)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class SGDFitter(BackendSGDFitter):
    """
    Fit population receptive field models with stochastic gradient descent (SGD).

    Estimates model parameters iteratively through SGD by minimizing the loss between model predictions and target
    data.

    Parameters
    ----------
    model : BaseModel
        Population receptive field model instance that can be fit to data.
        The model must implement `__call__` to make predictions that can be compared to data.
    stimulus : Stimulus
        Stimulus object used to make model predictions.
    adapter : Adapter, optional
        Adapter object to apply transformations to parameters during fitting.
    optimizer : keras.optimizers.Optimizer, optional
        Optimizer instance. Default is `None` where a `keras.optimizers.Adam` optimizer is used.
    loss : keras.optimizers.Loss or Callable, optional
        Loss instance or function with the signatur `f(y, y_pred)`, where `y` are the target data and `y_pred` are the
        model predicitons. Default is `None` where a `keras.optimizers.MeanSquaredError` loss is used.
    dtype : str, optional
        The dtype used for fitting. If `None` (the default), uses the dtype from
        :func:`prfmodel.utils.get_dtype`.

    Notes
    -----
    At each step during the fitting, the `model` makes a prediction for each batch in the target data
    given the `stimulus` and the current parameter values. The predictions are then compared to the target data and
    the parameter values are updated given the `optimizer` schedule.

    """

    def __init__(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        model: BaseModel,
        stimulus: PRFStimulus,
        adapter: Adapter | None = None,
        optimizer: keras.optimizers.Optimizer | None = None,
        loss: keras.losses.Loss | Callable | None = None,
        dtype: str | None = None,
    ):
        super().__init__()

        self.model = model
        self.stimulus = stimulus

        if adapter is None:
            adapter = Adapter()

        if optimizer is None:
            optimizer = keras.optimizers.Adam()

        if loss is None:
            loss = keras.losses.MeanSquaredError()

        self.adapter = adapter
        self.optimizer = optimizer
        self.loss = loss
        self.dtype = dtype

    def _create_variables(self, init_parameters: pd.DataFrame, fixed_parameters: Sequence[str]) -> None:
        # Keras automatically discovers variables stored in dicts and links to them in
        # 'self.trainable_variables' and 'self.non_trainable_variables'
        self._parameter_variables = {
            str(key): keras.Variable(val, dtype=self.dtype, name=key, trainable=key not in fixed_parameters)
            for key, val in init_parameters.items()
        }

    def _delete_variables(self) -> None:
        del self._parameter_variables

    def fit(
        self,
        data: Tensor,
        init_parameters: pd.DataFrame,
        fixed_parameters: Sequence[str] | None = None,
        num_steps: int = 1000,
    ) -> tuple[SGDHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : Tensor
            Target data to fit the model to. Must have shape (num_batches, num_frames), where `num_batches` is the
            number of batches for which parameters are estimated and `num_frames` is the number of time steps.
        init_parameters : pandas.DataFrame
            Dataframe with initial model parameters. Columns must contain different model parameters and
            rows parameter values for each batch in `data`.
        fixed_parameters : Sequence of str, optional
            Names of model parameters that are fixed to their starting values, i.e., their values are not optimized
            during the fitting. If `None` (the default), all parameters are optimized during fitting.
        num_steps : int, default=1000
            Number of optimization steps.

        Returns
        -------
        SGDHistory
            A history object that contains loss and metric values for each optimization step.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        if fixed_parameters is None:
            fixed_parameters = []

        # Initialize parameters on transformed scale
        init_parameters_transformed = self.adapter.transform(init_parameters)

        self._create_variables(init_parameters_transformed, fixed_parameters)

        self.optimizer.build(self.trainable_variables)

        self.compile(optimizer=self.optimizer, loss=self.loss)

        data = ops.convert_to_tensor(data, dtype=self.dtype)

        state = self._get_state()

        history = SGDHistory()

        with tqdm(range(num_steps)) as pbar:
            for step in pbar:
                logs, state = self._update_model_weights(self.stimulus, data, state)

                if logs:
                    display_logs = {}
                    for key, value in logs.items():
                        display_logs[key] = float(value)

                    pbar.set_postfix(display_logs)

                history.on_step_end(step, logs)

        if state is not None:
            trainable_variables, non_trainable_variables, _, metrics_variables = state
            for variable, value in zip(self.trainable_variables, trainable_variables, strict=False):
                variable.assign(value)
            for variable, value in zip(self.non_trainable_variables, non_trainable_variables, strict=False):
                variable.assign(value)

            num_metric_vars = 0

            for metric in self.metrics:
                this_metric_vars = metrics_variables[num_metric_vars : num_metric_vars + len(metric.variables)]
                for variable, value in zip(metric.variables, this_metric_vars, strict=False):
                    variable.assign(value)
                num_metric_vars += len(this_metric_vars)

        params = pd.DataFrame(
            {v.name: ops.convert_to_numpy(v.value) for v in self.trainable_variables + self.non_trainable_variables},
        )

        # Transform parameters back to natural scale
        params = self.adapter.inverse(params)

        self._delete_variables()

        # Sort result param columns according to initial parameter columns
        return history, params[init_parameters.columns]
