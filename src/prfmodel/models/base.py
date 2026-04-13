"""Model base classes."""

from abc import abstractmethod
from typing import Generic
from typing import TypeVar
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.stimuli import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol

S = TypeVar("S", bound=Stimulus)


class BaseResponse(ModelProtocol, Generic[S]):
    """
    Generic abstract base class for response models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom population receptive field models.
    Subclasses must override the abstract :meth:`__call__` method and must be defined
    with a specific stimulus type.

    """

    @doc
    @abstractmethod
    def __call__(self, stimulus: S, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus.

        Parameters
        ----------
        %(stimulus)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Model predictions of shape `(num_units, ...)` and dtype `dtype`. The number of units is the
            number of rows in `parameters`. The number and size of other axes depends on the stimulus.

        """


class BaseEncoder(ModelProtocol, Generic[S]):
    """
    Generic abstract base class for encoding model responses.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom encoding models.
    Subclasses must override the abstract :attr:`parameter_names` property and
    :meth:`__call__` method and must be defined with a specific stimulus type.

    """

    @doc
    @abstractmethod
    def __call__(
        self,
        stimulus: S,
        response: Tensor,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ):
        """Encode a model response with a stimulus.

        Parameters
        ----------
        %(stimulus)s
        response : :data:`prfmodel.typing.Tensor`
            Model response.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The stimulus encoded model response with shape `(num_units, ...)` dtype `dtype`. The number of units is
            the number of rows in :attr:`parameters`. The number and size of other axes depends on the stimulus and the
            response.

        """


class BaseComposite(ModelProtocol, Generic[S]):
    """
    Generic abstract base class for creating composite models.

    Cannot be instantiated on its own. Can only be used as a parent class to create custom composite models.
    Subclasses must override the abstract :meth:`__call__` method and must be defined
    with a specific stimulus type.
    This class is intended for combining multiple submodels into a composite model with a custom :meth:`__call__`
    method that defines how the submodels interact to make a composite prediction.

    Parameters
    ----------
    **models
        Submodels to be combined into the composite model. All submodel classes must inherit from
        :class:`~prfmodel.utils.ModelProtocol`.

    Raises
    ------
    TypeError
        If submodel classes do not inherit from :class:`~prfmodel.utils.ModelProtocol`.

    """

    def __init__(self, **models: ModelProtocol | None):
        super().__init__()

        for model in models.values():
            if model is not None and not isinstance(model, ModelProtocol):
                msg = "Model instance must implement the 'parameter_names' property"
                raise TypeError(msg)

        self.models = models

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters that are used by the submodels."""
        param_names = []

        for model in self.models.values():
            if model is not None:
                param_names.extend(model.parameter_names)

        # Make sure no duplicates are returned (preserve insertion order)
        return list(dict.fromkeys(param_names))

    @doc
    @abstractmethod
    def __call__(
        self,
        stimulus: S,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a composite model response to a stimulus.

        Parameters
        ----------
        %(stimulus)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
