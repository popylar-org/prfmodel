"""Abstract base classes for scaling models.

Classes in this module inherit from :class:`~prfmodel.protocols.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.protocols.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example, :class:`~prfmodel.models.base.BaseScaling` defines that all child classes
must implement a :meth:`~prfmodel.models.base.BaseScaling.call` method that takes a model response and a set of
parameters as input. However, it leaves it up to each child class to define how input response parameters are used to
make model predictions.

All base classes have a concrete user-facing :meth:``__call__`` method
(e.g., :meth:`~prfmodel.scaling.base.BaseScaling.__call__`) that takes non-tensor arguments and
performs validation checks. This method calls the abstract ``call`` method that must be implemented by each child
class and only accepts tensor arguments to enable backend compilation.

"""

from abc import abstractmethod
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.protocols import ModelProtocol
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import as_tensor_frame
from prfmodel.utils import get_dtype


class BaseScaling(ModelProtocol):
    """
    Abstract base class for scaling models.

    Scaling models modify a temporal input response (e.g., a neural response convolved with an impulse response).

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom response
    models. Subclasses must override the abstract :attr:`parameter_names` property and the :meth:`call` method.
    Do not override :meth:`__call__`; it is the public facade that validates and converts before delegating to
    :meth:`call`.

    """

    @doc
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Make predictions with the scaling model.

        This is the public entry point; subclasses implement :meth:`call` instead.

        Parameters
        ----------
        inputs : :data:`prfmodel.typing.Tensor`
            Input tensor with temporal response and shape (num_units, num_frames).
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        Raises
        ------
        %(raises_missing_parameters)s

        """
        dtype = get_dtype(dtype)
        self.check_parameter_names(parameters)
        self.check_parameter_values(parameters)

        return self.call(
            ops.convert_to_tensor(inputs, dtype=dtype),
            as_tensor_frame(parameters[self.parameter_names], dtype),
        )

    @doc
    @abstractmethod
    def call(self, inputs: Tensor, parameters: TensorFrame) -> Tensor:
        """
        Make predictions with the scaling model, from tensors.

        Parameters
        ----------
        inputs : :data:`prfmodel.typing.Tensor`
            Input tensor with temporal response and shape (num_units, num_frames).
        %(parameters_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        Notes
        -----
        Implementations must be traceable by a backend compiler. See
        :meth:`~prfmodel.models.base.BasePopulationResponse.call`.

        """
