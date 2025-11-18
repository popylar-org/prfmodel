"""JAX fitter implementations."""

import jax
import jax.numpy as jnp
import keras
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict
from .base import BaseSGDFitter
from .base import SGDState


class JAXSGDFitter(BaseSGDFitter):
    """JAX stochastic gradient descent fitter."""

    def _compute_loss_and_updates(
        self,
        trainable_variables: list,
        non_trainable_variables: list,
        x: Stimulus,
        y: Tensor,
        dtype: str | None,
    ) -> tuple[Tensor, tuple[Tensor, list]]:
        state_mapping: list[tuple[str, Tensor]] = []
        state_mapping.extend(zip(self.trainable_variables, trainable_variables, strict=False))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables, strict=False))

        with keras.StatelessScope(state_mapping) as scope:
            params = ParamsDict(
                {
                    key.name: val
                    for key, val in zip(
                        self.trainable_variables + self.non_trainable_variables,
                        trainable_variables + non_trainable_variables,
                        strict=False,
                    )
                },
            )
            # Make model predictions with parameters on natural scale
            params = self.adapter.inverse(params)
            y_pred = self.model(x, params, dtype=dtype)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        non_trainable_variables = [scope.get_current_value(v) for v in self.non_trainable_variables]

        return loss, (y_pred, non_trainable_variables)

    def _get_state(self) -> SGDState:
        return self.trainable_variables, self.non_trainable_variables, self.optimizer.variables, self.metrics_variables

    def _update_model_weights(self, x: Stimulus, y: Tensor, state: SGDState) -> tuple[dict, SGDState]:
        if state is None:
            msg = "State must not be None when using JAX backend"
            raise TypeError(msg)

        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state

        grad_fn = jax.value_and_grad(self._compute_loss_and_updates, has_aux=True)

        trainable_variables = [jnp.array(v, dtype=self.dtype) for v in trainable_variables]
        non_trainable_variables = [jnp.array(v, dtype=self.dtype) for v in non_trainable_variables]

        (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            x,
            y,
            self.dtype,
        )

        (
            trainable_variables,
            optimizer_variables,
        ) = self.optimizer.stateless_apply(
            optimizer_variables,
            grads,
            trainable_variables,
        )

        new_metrics_vars: list[keras.metrics.Metric] = []
        logs = {}

        for metric in self.metrics:
            this_metric_vars = metrics_variables[len(new_metrics_vars) : len(new_metrics_vars) + len(metric.variables)]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars,
                    y,
                    y_pred,
                )
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )

        return logs, state
