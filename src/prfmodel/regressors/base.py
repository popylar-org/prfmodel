"""Abstract base class for regressor models.

Classes in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example, :class:`~prfmodel.regressors.base.BaseRegressors` defines that all child
classes must implement a :meth:`~prfmodel.regressors.base.BaseRegressors.call` method that takes a set of regressors
and a set of parameters as input. However, it leaves it up to each child class to define how regresors and
parameters are used to make model predictions.

All base classes have a concrete user-facing :meth:``__call__`` method
(e.g., :meth:`~prfmodel.regressors.base.BaseRegressors.__call__`) that takes non-tensor arguments and
performs validation checks. This method calls the abstract ``call`` method that must be implemented by each child
class and only accepts tensor arguments to enable backend compilation.

Regressor models contribute an additive linear term to a canonical model prediction. Each regressor is a fixed
time course (column of a design matrix) that is multiplied by a per-unit beta weight. Concrete subclasses define
how the regressor design matrix is transformed before being weighted (e.g., whether it is convolved with an
impulse response).

The regressor design data is supplied at call time as a :class:`pandas.DataFrame` or
:class:`~prfmodel.utils.TensorFrame` whose columns include (at least) the regressor names.
Column order is unimportant and extra columns are silently ignored.

"""

from abc import abstractmethod
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import CompositeModelProtocol
from prfmodel.utils import TensorFrame
from prfmodel.utils import as_tensor_frame
from prfmodel.utils import get_dtype


def _normalize_regressors_model(
    regressors_model: "BaseRegressors | list[BaseRegressors] | None",
) -> "BaseRegressors | None":
    """Wrap a list of regressor models in a :class:`RegressorsList`; pass other values through unchanged.

    Helper shared by canonical models to accept either a single regressor model, a list of regressor models, or
    ``None`` for the ``regressors_model`` constructor argument.

    """
    from ._list import RegressorsList  # noqa: PLC0415 (local import avoids a circular dependency)

    if isinstance(regressors_model, list):
        return RegressorsList(regressors_model)
    return regressors_model


def _validate_regressors_argument(
    regressors_model: object | None,
    regressors: pd.DataFrame | None,
) -> None:
    """Validate the ``regressors`` argument against a model's configured ``regressors_model``.

    Helper shared by canonical models and fitters to ensure that runtime regressor design data is supplied if (and
    only if) a regressors model is configured, and that the design it carries covers the columns that model needs.

    The design check runs here rather than only in :meth:`BaseRegressors.__call__` because a composite model and a
    fitter both reach the regressors model through :meth:`BaseRegressors.call`, where a missing column would
    surface as a ``KeyError`` from inside a trace instead of as a clear message.

    Raises
    ------
    ValueError
        If ``regressors`` is provided without a configured ``regressors_model``, if a ``regressors_model`` is
        configured but ``regressors`` is not provided, or if the design is missing a required column.

    """
    if regressors_model is None and regressors is not None:
        msg = "'regressors' was provided but 'regressors_model' is not configured on this model"
        raise ValueError(msg)
    if regressors_model is not None and regressors is None:
        msg = "'regressors' must be provided when 'regressors_model' is configured on this model"
        raise ValueError(msg)
    if isinstance(regressors_model, BaseRegressors) and regressors is not None:
        regressors_model.check_regressor_names(regressors)


def _extract_regressor_design(
    regressors_model: "BaseRegressors | None",
    regressors: pd.DataFrame | None,
    dtype: str,
) -> "TensorFrame | None":
    """Convert the design columns a regressors model reads into tensors.

    Returns ``None`` when there is no design to convert. Columns outside the model's
    :attr:`BaseRegressors.regressor_names` are left behind, so a caller may carry extra columns -- including
    non-numeric ones -- alongside the design in the same frame.

    """
    if regressors is None or regressors_model is None:
        return None

    return as_tensor_frame(regressors[regressors_model.regressor_names], dtype)


class BaseRegressors(CompositeModelProtocol):
    r"""
    Abstract base class for regressor models.

    A regressor model returns the additive contribution
    :math:`\sum_k \beta_k \, x_k(t)` of a set of regressors :math:`x_k(t)` to a model prediction. The per-unit
    weights :math:`\beta_k` come from the ``parameters`` argument and the design columns :math:`x_k(t)` come from
    the ``regressors`` data frame at call time.

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom regressor
    models. Subclasses must override the abstract :meth:`call` method. Child classes should either assign the
    :attr:`_regressor_names` attribute during initialization or overwrite the :meth:`regressor_names` property and
    the :meth:`check_regressor_names` method.

    """

    _regressor_names: tuple[str, ...] = ()
    """Design columns this model reads itself, on top of the ones its submodels read."""

    @property
    def regressor_names(self) -> list[str]:
        """Columns this model reads from the regressor design."""
        return list(self._regressor_names)

    def check_regressor_names(self, regressors: pd.DataFrame) -> None:
        """Check that required columns are supplied in the regressor design.

        Parameters
        ----------
        %(regressors)s

        Raises
        ------
        ValueError
            When a column is missing in the regressor design.

        """
        missing = [name for name in self.regressor_names if name not in regressors.columns]

        if missing:
            msg = f"Regressor design is missing required column(s): {missing}"
            raise ValueError(msg)

    @doc
    def __call__(
        self,
        regressors: pd.DataFrame,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Compute the additive regressor contribution.

        This is the public entry point; subclasses implement :meth:`call` instead.

        Parameters
        ----------
        %(regressors)s
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
        self.check_regressor_names(regressors)

        return self.call(
            as_tensor_frame(regressors[self.regressor_names], dtype),
            as_tensor_frame(parameters[self.get_consumed_parameter_names(parameters)], dtype),
        )

    @doc
    @abstractmethod
    def call(self, regressors: TensorFrame, parameters: TensorFrame) -> Tensor:
        """
        Compute the additive regressor contribution from tensors.

        Parameters
        ----------
        %(regressors_tensors)s
        %(parameters_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        Notes
        -----
        Implementations must be traceable by a backend compiler. See
        :meth:`~prfmodel.models.base.BasePopulationResponse.call`.

        """
