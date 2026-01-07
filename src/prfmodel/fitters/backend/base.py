"""Backend-specific fitter base classes."""

from abc import abstractmethod
from typing import TypeAlias
import keras
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype

SGDState: TypeAlias = tuple[list, list, list, list] | None
"""State with objects that are carried during the stochastic gradient descent optimization."""


class BaseSGDFitter(keras.Model):
    """Backend-specific stochastic gradient descent base class."""

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

    @abstractmethod
    def _get_state(self) -> SGDState:
        pass

    @abstractmethod
    def _update_model_weights(self, x: Stimulus, y: Tensor, state: SGDState) -> tuple[dict, SGDState]:
        pass
