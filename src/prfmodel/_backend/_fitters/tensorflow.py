"""Tensorflow fitter implementations."""

import tensorflow as tf
from prfmodel.stimuli import StimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict
from .base import BaseSGDFitter
from .base import SGDState


class TensorFlowSGDFitter(BaseSGDFitter):
    """Tensorflow stochastic gradient descent fitter.

    Notes
    -----
    When `compile_step` is enabled, the optimization step is wrapped in ``tf.function`` to run it in graph
    execution mode. The step reads and writes Keras variables in place, so the state passed in and out is
    `None` and the traced function takes no tensor arguments at all. To force eager execution without changing
    `compile_step`, use ``tf.config.run_functions_eagerly(True)``.

    """

    def _get_state(self) -> SGDState:
        return None

    def _update_model_weights(
        self,
        x: StimulusTensors,
        y: Tensor,
        state: SGDState,
        regressors: ParamsDict | None,
    ) -> tuple[dict, SGDState]:
        with tf.GradientTape() as tape:
            # Important to create this inside gradient tape because we transform keras variables
            params = ParamsDict(
                {v.name: v.value for v in self.trainable_variables + self.non_trainable_variables},
                dtype=self.dtype,
            )
            # Make model predictions with parameters on natural scale
            params = self.adapter.inverse(params)
            y_pred = self.model.call(x, params, regressors=regressors)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply(gradients, self.trainable_variables)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        return logs, state
