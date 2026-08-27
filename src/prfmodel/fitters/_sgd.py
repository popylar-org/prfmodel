"""Stochastic gradient descent fitters."""

from collections.abc import Callable
from collections.abc import Sequence
import keras
import pandas as pd
from keras import ops
from tqdm.auto import tqdm
from prfmodel._backend._fitters import BackendSGDFitter
from prfmodel._docstring import doc
from prfmodel.fitters.adapter import Adapter
from prfmodel.models.base import BaseCanonical
from prfmodel.stimuli import Stimulus
from prfmodel.typing import Tensor


class SGDHistory:
    """Stochastic gradient descent metric history.

    Logs losses and metrics over stochastic gradient descent steps.

    Attributes
    ----------
    history : dict
        Dictionary with keys indicating metric names and values containing metric values for each step.
    step : list of int
        List of step indices. When fitting is split into data batches (see `batch_size` in :meth:`SGDFitter.fit`),
        each batch is optimized independently but its steps continue counting up from where the previous batch left
        off, so `step` and `history` simply grow longer rather than changing meaning.

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


@doc
class SGDFitter(BackendSGDFitter):
    """
    Fit population receptive field models with stochastic gradient descent (SGD).

    Estimates model parameters iteratively through SGD by minimizing the loss between model predictions and target
    data.

    Parameters
    ----------
    %(model_fitter)s
    %(stimulus)s
    adapter : Adapter, optional
        Adapter object to apply transformations to parameters during fitting.
    optimizer : keras.optimizers.Optimizer, optional
        Optimizer instance. Default is `None` where a `keras.optimizers.Adam` optimizer is used.
    loss : keras.optimizers.Loss or Callable, optional
        Loss instance or function with the signatur `f(y, y_pred)`, where `y` are the target data and `y_pred` are the
        model predicitons. Default is `None` where a `keras.optimizers.MeanSquaredError` loss is used.
    compile_step : bool, default=False
        Whether to compile the optimization step with the backend's native primitive: `jax.jit` on JAX,
        `tf.function` on TensorFlow and `torch.compile` on PyTorch. With `False` (the default) the step
        runs eagerly, which is slower but makes the step inspectable and produces Python
        tracebacks that point at the offending line. Whether compiling pays off depends on the backend
        and on the problem size.
    %(dtype)s

    Notes
    -----
    At each step during the fitting, the `model` makes a prediction for each batch in the target data
    given the `stimulus` and the current parameter values. The predictions are then compared to the target data and
    the parameter values are updated given the `optimizer` schedule.

    When `batch_size` is set in :meth:`fit`, units are split into data batches that are each optimized independently
    for `num_steps` steps, with freshly-initialized optimizer state per batch. This bounds memory use for large
    datasets without changing the result for any given unit, since units do not share parameters or gradients.

    When `compile_step` is enabled, the optimization step is compiled once per data batch and then reused
    for every step, so the cost of tracing is paid once rather than `num_steps` times. Compiling requires
    that no tensor value is read back inside the step, since while the graph is being built there are no
    values to read. Parameter checks therefore run before the loop: the initial parameters are passed
    through one prediction while nothing is traced yet, so invalid starting values still raise with a
    useful message. Values that only become invalid *during* the optimization are not reported and show
    up as a `NaN` loss instead.

    Examples
    --------
    Fit a 2D Gaussian population receptive field model.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models.prf import Gaussian2DPRFModel
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> # Only fit response model
    >>> model = Gaussian2DPRFModel(
    ...     impulse_model=None,
    ...     scaling_model=None,
    ... )
    >>> fitter = SGDFitter(model=model, stimulus=stimulus)
    >>> # Define start parameters
    >>> params_init = pd.DataFrame({
    ...     "mu_y": [0.0],
    ...     "mu_x": [0.0],
    ...     "sigma": [1.0],
    ... })
    >>> # Create dummy data for a single unit
    >>> data = np.zeros((1, stimulus.design.shape[0]))
    >>> # Fit model parameters
    >>> history, params_sgd = fitter.fit(data, params_init, num_steps=5)
    >>> print(list(params_sgd.columns))
    ['mu_y', 'mu_x', 'sigma']

    """

    @doc
    def __init__(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        model: BaseCanonical,
        stimulus: Stimulus,
        adapter: Adapter | None = None,
        optimizer: keras.optimizers.Optimizer | None = None,
        loss: keras.losses.Loss | Callable | None = None,
        compile_step: bool = False,
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
        self.compile_step = compile_step
        self.dtype = dtype

    def _create_variables(self, init_parameters: pd.DataFrame, fixed_parameters: Sequence[str]) -> None:
        # Keras automatically discovers variables stored in dicts and links to them in
        # 'self.trainable_variables' and 'self.non_trainable_variables'
        #
        # Only the columns the model reads become variables. A column no model consumes cannot move the
        # prediction, so there is nothing to optimize, and requiring it to be numeric would break the
        # contract that lets a caller carry a label alongside the parameters in one frame.
        self._parameter_variables = {
            str(key): keras.Variable(val, dtype=self.dtype, name=key, trainable=key not in fixed_parameters)
            for key, val in init_parameters.items()
            if key in self.model.parameter_names
        }

    def _delete_variables(self) -> None:
        del self._parameter_variables

    def fit(  # noqa: PLR0913 (too many arguments in function definition)
        self,
        data: Tensor,
        init_parameters: pd.DataFrame,
        fixed_parameters: Sequence[str] | None = None,
        num_steps: int = 1000,
        batch_size: int | None = None,
        regressors: pd.DataFrame | None = None,
    ) -> tuple[SGDHistory, pd.DataFrame]:
        """
        Fit a population receptive field model to target data.

        Parameters
        ----------
        data : :data:`prfmodel.typing.Tensor`
            Target data to fit the model to. Must have shape `(num_units, num_frames)`, where `num_units` is the
            number of units for which parameters are estimated and `num_frames` is the number of time steps.
        init_parameters : pandas.DataFrame
            Dataframe with initial model parameters. Columns must contain different model parameters and
            rows parameter values for each unit in `data`.
        fixed_parameters : Sequence of str, optional
            Names of model parameters that are fixed to their starting values, i.e., their values are not optimized
            during the fitting. If `None` (the default), all parameters are optimized during fitting.
        num_steps : int, default=1000
            Number of optimization steps.
        batch_size : int, optional
            Number of units to fit at the same time. If `None` (the default), all units are fit at once.
        regressors : pandas.DataFrame, optional
            Runtime regressor design data forwarded to every model prediction call. Must be provided when the model
            has a ``regressors_model`` configured.

        Returns
        -------
        SGDHistory
            A history object that contains loss and metric values for each optimization step.
        pandas.DataFrame
            A dataframe with final model parameters.

        """
        if fixed_parameters is None:
            fixed_parameters = []

        num_units = len(init_parameters)
        if batch_size is None:
            batch_size = num_units

        history = SGDHistory()
        param_batches = []
        step_offset = 0

        batch_starts = range(0, num_units, batch_size)
        for start in tqdm(batch_starts, desc="Processing data batches", total=len(batch_starts)):
            end = min(start + batch_size, num_units)

            # Each batch is fit on a freshly-constructed instance rather than reusing 'self'. Reusing
            # 'self' across batches would keep accumulating every previous batch's parameter variables
            # onto this instance's trainable_variables.
            batch_fitter = type(self)(
                model=self.model,
                stimulus=self.stimulus,
                adapter=self.adapter,
                optimizer=keras.optimizers.deserialize(keras.optimizers.serialize(self.optimizer)),
                loss=self.loss,
                compile_step=self.compile_step,
                dtype=self.dtype,
            )
            batch_history, batch_params = batch_fitter._fit_batch(  # noqa: SLF001 (private member access)
                data[start:end],
                init_parameters.iloc[start:end],
                fixed_parameters,
                num_steps,
                regressors,
            )

            for step in batch_history.step:
                history.step.append(step_offset + step)
            for key, values in batch_history.history.items():
                history.history.setdefault(key, []).extend(values)
            step_offset += len(batch_history.step)

            param_batches.append(batch_params)

        new_parameters = pd.concat(param_batches, ignore_index=True)
        new_parameters.index = init_parameters.index

        return history, new_parameters

    def _fit_batch(
        self,
        data_batch: Tensor,
        init_parameters_batch: pd.DataFrame,
        fixed_parameters: Sequence[str],
        num_steps: int,
        regressors: pd.DataFrame | None,
    ) -> tuple[SGDHistory, pd.DataFrame]:
        """Fit a single data batch and return its step history and final parameters."""
        # Initialize parameters on transformed scale
        init_parameters_transformed = self.adapter.transform(init_parameters_batch)

        self._create_variables(init_parameters_transformed, fixed_parameters)

        self.optimizer.build(self.trainable_variables)

        self.compile(optimizer=self.optimizer, loss=self.loss)

        data_batch = ops.convert_to_tensor(data_batch, dtype=self.dtype)

        state = self._get_state()

        history = SGDHistory()

        # One validated prediction before the loop, so that starting values outside a model's domain are
        # reported with the message the model raises rather than as a NaN loss thousands of steps later.
        y_pred = self.model(
            self.stimulus,
            self.adapter.inverse(init_parameters_transformed),
            regressors=regressors,
            dtype=self.dtype,
        )

        # Keras builds the compiled loss on its first call. Doing that here rather than inside the step
        # keeps the build out of the traced region, where TorchDynamo fails to rewrite it. The call has no
        # side effect beyond the build: it updates no metric and touches no variable.
        self.compute_loss(y=data_batch, y_pred=y_pred)

        step_fun = self._make_step_fun(self.stimulus, data_batch, regressors)

        # Reading a log value back from the device synchronizes with it, which would serialize the whole
        # loop against the progress bar. Refreshing about a hundred times is enough for a progress bar.
        postfix_every = max(1, num_steps // 100)

        with tqdm(range(num_steps), desc="Optimizing batch", leave=False) as pbar:
            for step in pbar:
                logs, state = step_fun(state)

                if logs and step % postfix_every == 0:
                    pbar.set_postfix({key: float(value) for key, value in logs.items()})

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

        # Transform parameters back to natural scale, sorted according to the initial parameter columns.
        # Columns the model does not read never became variables, so they are carried over from the input
        # unchanged instead of being looked up among the estimates. Values are taken as an array because the
        # estimates carry a fresh index that need not line up with the batch's.
        params = self.adapter.inverse(params)

        for name in init_parameters_batch.columns:
            if name not in params.columns:
                params[name] = init_parameters_batch[name].to_numpy()

        params = params[init_parameters_batch.columns]

        self._delete_variables()

        return history, params
