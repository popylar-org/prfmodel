"""PyTorch fitter implementations."""

import torch
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict
from .base import BaseSGDFitter
from .base import SGDState


class TorchSGDFitter(BaseSGDFitter):
    """PyTorch stochastic gradient descent fitter."""

    def _get_state(self) -> SGDState:
        return None

    def _update_model_weights(self, x: Stimulus, y: Tensor, state: SGDState) -> tuple[dict, SGDState]:
        self.zero_grad()

        params = ParamsDict({v.name: v.value for v in self.trainable_variables + self.non_trainable_variables})
        # Make model predictions with parameters on natural scale
        params = self.adapter.inverse(params)

        y_pred = self.model(x, params, dtype=self.dtype)

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
