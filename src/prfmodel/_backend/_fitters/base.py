"""Backend-specific fitter base classes."""

from abc import abstractmethod
from collections.abc import Callable
from typing import TypeAlias
from typing import cast
import keras
import pandas as pd
from prfmodel._backend import compile_fun
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _extract_regressor_design
from prfmodel.stimuli import Stimulus
from prfmodel.stimuli import StimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import get_dtype

SGDState: TypeAlias = tuple[list, list, list, list] | None
"""State with objects that are carried during the stochastic gradient descent optimization."""

StepFun: TypeAlias = Callable[[SGDState], tuple[dict, SGDState]]
"""Optimization step function that maps the incoming state onto step metrics and the outgoing state."""


class BaseSGDFitter(keras.Model):
    """Backend-specific stochastic gradient descent base class."""

    compile_step: bool = False
    """Whether the optimization step is compiled."""

    @property
    def dtype(self) -> str:
        """
        The dtype that is used during fitting.

        If `None`, uses `keras.config.floatx()` which defaults
        to `float32`.
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: str | None) -> None:
        self._dtype = get_dtype(value)

    def _make_step_fun(self, x: Stimulus, y: Tensor, regressors: pd.DataFrame | None) -> StepFun:
        """Build the function that performs a single optimization step.

        The stimulus, the target data and the regressors are the same on every step, so they are converted to
        tensors once here and captured, instead of being passed and reconverted per step. That leaves the
        optimization state as the only argument, and the state is made of tensors alone.

        Backends differ in *how* they carry state: JAX threads it through the returned `SGDState` while
        TensorFlow and PyTorch mutate Keras variables in place and return `None`. This difference
        is implemented in :meth:`_get_state` and :meth:`_update_model_weights`. The compilation primitive itself is
        resolved by :func:`prfmodel._backend.compile_fun`, so this implementation is shared.

        """
        stimulus = x.to_tensors(self.dtype)
        regressors_model = cast("BaseRegressors | None", self.model.models.get("regressors_model"))
        regressors_consumed = _extract_regressor_design(regressors_model, regressors, self.dtype)

        def step(state: SGDState) -> tuple[dict, SGDState]:
            return self._update_model_weights(stimulus, y, state, regressors_consumed)

        return compile_fun(step) if self.compile_step else step

    @abstractmethod
    def _get_state(self) -> SGDState:
        pass

    @abstractmethod
    def _update_model_weights(
        self,
        x: StimulusTensors,
        y: Tensor,
        state: SGDState,
        regressors: TensorFrame | None,
    ) -> tuple[dict, SGDState]:
        pass
