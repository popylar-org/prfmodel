"""PyTorch fitter implementations."""

import torch
from prfmodel.stimuli import StimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from .base import BaseSGDFitter
from .base import SGDState


class TorchSGDFitter(BaseSGDFitter):
    """PyTorch stochastic gradient descent fitter.

    Notes
    -----
    When `compile_step` is enabled, the optimization step is wrapped in ``torch.compile``. The step reads and
    writes Keras variables in place, so the state passed in and out is `None` and the compiled function takes
    no tensor arguments at all.

    """

    def _get_state(self) -> SGDState:
        return None

    def _update_model_weights(
        self,
        x: StimulusTensors,
        y: Tensor,
        state: SGDState,
        regressors: TensorFrame | None,
    ) -> tuple[dict, SGDState]:
        self.zero_grad()

        params = TensorFrame(
            {v.name: v.value for v in self.trainable_variables + self.non_trainable_variables},
            dtype=self.dtype,
        )
        # Make model predictions with parameters on natural scale
        params = self.adapter.inverse(params)

        y_pred = self.model.call(x, params, regressors=regressors)

        loss = self.compute_loss(y=y, y_pred=y_pred)

        loss.backward()

        gradients = [v.value.grad for v in self.trainable_weights]

        with torch.no_grad():
            self.optimizer.apply(gradients, self.trainable_weights)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        logs = {m.name: m.result() for m in self.metrics}

        return logs, state
