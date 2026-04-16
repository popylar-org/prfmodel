"""Abstract base classes for scaling models.

Classes in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example, :class:`~prfmodel.models.base.BaseScaling` defines that all child classes
must implement a :meth:`~prfmodel.models.base.BaseScaling.__call__` method that takes a model response and a set of
parameters as input. However, it leaves it up to each child class to define how input response parameters are used to
make model predictions.

"""

from abc import abstractmethod
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol


class BaseScaling(ModelProtocol):
    """
    Abstract base class for scaling models.

    Scaling models modify a temporal input response (e.g., a neural response convolved with an impulse response).

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom response
    models. Subclasses must override the abstract :attr:`parameter_names` and :meth:`__call__` method.

    """

    @doc
    @abstractmethod
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Make predictions with the scaling model.

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
