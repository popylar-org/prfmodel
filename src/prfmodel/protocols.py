"""Protocol classes for models and composite models.

Classes in this module define the interface that all model classes share: a
:attr:`~prfmodel.protocols.ModelProtocol.parameter_names` property and the validation checks that are performed on
user-supplied parameters. :class:`~prfmodel.protocols.CompositeModelProtocol` extends this interface to models that
are composed of named submodels and forwards the validation checks to each of them.

"""

from abc import abstractmethod
from collections.abc import Generator
from collections.abc import Sequence
from typing import ClassVar
from typing import Protocol
from typing import runtime_checkable
import pandas as pd
from ._docstring import doc


def _check_parameter_names(parameter_names: Sequence[str], parameters: pd.DataFrame) -> None:
    missing_params = [param for param in parameter_names if param not in parameters.columns]
    if missing_params:
        msg = f"Missing required parameter names: {missing_params}"
        raise ValueError(msg)


def _check_parameter_values(parameter_names: Sequence[str], parameters: pd.DataFrame) -> None:
    for name in parameter_names:
        if name in parameters.columns and not (parameters[name].to_numpy() > 0.0).all():
            msg = f"Parameter '{name}' must be > 0"
            raise ValueError(msg)


@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol for model classes.

    Cannot be instantiated on its own.
    This protocol is intended as the parent class for models that can be composed with
    :class:`~prfmodel.protocols.CompositeModelProtocol`.
    Subclasses must override the abstract :attr:`parameter_names` property.

    Attributes
    ----------
    parameter_names : list of str
        Names of parameters used by the model class.

    Examples
    --------
    Create a custom model class that inherits from the protocol and overwrites the :attr:`parameter_names` property.

    >>> class CustomModel(ModelProtocol):
    ...     @property
    ...     def parameter_names(self):
    ...         return ["a", "b"]
    >>> model = CustomModel()
    >>> print(model.parameter_names)
    ['a', 'b']

    """

    _positive_parameter_names: ClassVar[tuple[str, ...]] = ()
    """Parameters that must be strictly positive for the model to be defined."""

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """A list with names of parameters that are used by the model."""

    @doc
    def check_parameter_names(self, parameters: pd.DataFrame) -> None:
        """Check that required parameter names are supplied.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter name in the :attr:`parameter_names` attribute is not a column in ``parameters``.

        """
        _check_parameter_names(self.parameter_names, parameters)

    @doc
    def check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the model is defined on.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter that must be ``> 0`` is zero or negative.

        """
        _check_parameter_values(list(self._positive_parameter_names), parameters)

    @doc
    def get_consumed_parameter_names(self, parameters: pd.DataFrame) -> list[str]:
        """Return the parameter names the model reads from ``parameters``.

        A name covered by :attr:`default_parameters` is only read when the caller supplies a column for it;
        otherwise the default is merged in further down and the column would be absent here.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        list of str
            Names of the parameters the model reads from ``parameters``.

        """
        default_parameters = getattr(self, "default_parameters", None) or {}

        return [name for name in self.parameter_names if name not in default_parameters or name in parameters.columns]


class CompositeModelProtocol(ModelProtocol):
    """Protocol for composite model classes.

    Composite model classes contain one or multiple submodels that are stored as a dict ``{"name": model_instance}``
    in the private :attr:`_models` attribute. This class forwards parameter name and value checks to each submodel. It
    is intended as a parent class for canonical models, e.g., :class:`~prfmodel.models.base.BaseCanonical`.
    Child classes should define how submodels are stored in
    :attr:`_models` by using the public setter method (see example).

    Optionally, child classes can define :attr:`_additional_parameter_names` to add parameters that are not part of
    any submodel but still required by the model.

    Examples
    --------
    Create a custom composite model with a single submodel.

    >>> class DummyModel(ModelProtocol):
    ...     @property
    ...     def parameter_names(self):
    ...         return ["a"]
    >>> class CompositeDummyModel(CompositeModelProtocol):
    ...      def __init__(self, dummy_model):
    ...         self.models = {"dummy_model": dummy_model}  # Use the public setter to store the submodels
    >>> model = DummyModel()
    >>> composite_model = CompositeDummyModel(dummy_model=model)
    >>> composite_model.parameter_names  # The composite model collects the parameters of all submodels
    ['a']

    """

    _additional_parameter_names: tuple[str, ...] = ()
    """Parameters this model reads itself, on top of the ones its submodels read."""

    def __init__(self) -> None:
        self._models: dict[str, ModelProtocol | None] = {}

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters that are used by the submodels."""
        param_names = list(self._additional_parameter_names)

        for model in self._iter_models():
            param_names.extend(model.parameter_names)

        # Make sure no duplicates are returned (preserve insertion order)
        return list(dict.fromkeys(param_names))

    @property
    def models(self) -> dict[str, ModelProtocol | None]:
        """A dictionary with the named submodels.

        Parameters
        ----------
        models: dict of ModelProtocol
            Named submodels.

        Raises
        ------
        TypeError
            When a submodel does not inherit from :class:`ModelProtocol`.

        """
        return self._models

    @models.setter
    def models(self, models: dict[str, ModelProtocol | None]) -> None:
        for model in models.values():
            if model is not None and not isinstance(model, ModelProtocol):
                msg = "Models must inherit from 'prfmodel.protocols.ModelProtocol'"
                raise TypeError(msg)

        self._models = models

    def _iter_models(self) -> Generator[ModelProtocol]:
        for model in self.models.values():
            if model is not None:
                yield model

    @doc
    def check_parameter_names(self, parameters: pd.DataFrame) -> None:
        """Check that required parameter names are supplied.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter name in the :attr:`parameter_names` attribute is not a column in ``parameters``.

        """
        # Check own parameters
        _check_parameter_names(self._additional_parameter_names, parameters)

        # Check parameters of each submodel
        for model in self._iter_models():
            model.check_parameter_names(parameters)

    @doc
    def check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the model is defined on.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter that must be ``> 0`` is zero or negative.

        """
        _check_parameter_values(list(self._positive_parameter_names), parameters)

        for model in self._iter_models():
            model.check_parameter_values(parameters)

    @doc
    def get_consumed_parameter_names(self, parameters: pd.DataFrame) -> list[str]:
        """Return the parameter names this model and its submodels read from ``parameters``.

        Parameters
        ----------
        %(parameters)s

        Returns
        -------
        list of str
            Names of the parameters this model and its submodels read from ``parameters``.

        """
        consumed_param_names = list(self._additional_parameter_names)

        for model in self._iter_models():
            consumed_param_names.extend(model.get_consumed_parameter_names(parameters))

        return list(dict.fromkeys(consumed_param_names))
