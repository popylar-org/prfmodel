"""Abstract base classes for scaling models."""

from abc import abstractmethod
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol


class BaseTemporal(ModelProtocol):
    """
    Abstract base class for temporal models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom temporal models.
    Subclasses must override the abstract :meth:`__call__` method.

    """

    @doc
    @abstractmethod
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Make predictions with the temporal model.

        Parameters
        ----------
        inputs : :data:`prfmodel.typing.Tensor`
            Input tensor with temporal response and shape (num_units, num_frames).
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
