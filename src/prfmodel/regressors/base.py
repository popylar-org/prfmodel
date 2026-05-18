"""Abstract base class for regressor models.

Classes in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

Regressor models contribute an additive linear term to a canonical model prediction. Each regressor is a fixed
time course (column of a design matrix) that is multiplied by a per-unit beta weight. Concrete subclasses define
how the regressor design matrix is transformed before being weighted (e.g., whether it is convolved with an
impulse response).

The regressor design data is supplied at call time as a :class:`pandas.DataFrame` whose columns include (at
least) the regressor names. Column order is unimportant and extra columns are silently ignored.

"""

from abc import abstractmethod
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol


def _extract_design(regressors: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Select the required regressor columns from a design DataFrame.

    Raises a clear ``ValueError`` if any required column is missing.

    """
    missing = [name for name in names if name not in regressors.columns]
    if missing:
        msg = f"Regressor design is missing required column(s): {missing}"
        raise ValueError(msg)
    return regressors[names]


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

    Helper shared by canonical models to ensure that runtime regressor design data is supplied if (and only if) a
    regressors model is configured.

    Raises
    ------
    ValueError
        If ``regressors`` is provided without a configured ``regressors_model``, or if a ``regressors_model`` is
        configured but ``regressors`` is not provided.

    """
    if regressors_model is None and regressors is not None:
        msg = "'regressors' was provided but 'regressors_model' is not configured on this model"
        raise ValueError(msg)
    if regressors_model is not None and regressors is None:
        msg = "'regressors' must be provided when 'regressors_model' is configured on this model"
        raise ValueError(msg)


class BaseRegressors(ModelProtocol):
    r"""
    Abstract base class for regressor models.

    A regressor model returns the additive contribution
    :math:`\sum_k \beta_k \, x_k(t)` of a set of regressors :math:`x_k(t)` to a model prediction. The per-unit
    weights :math:`\beta_k` come from the ``parameters`` argument and the design columns :math:`x_k(t)` come from
    the ``regressors`` data frame at call time.

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom regressor
    models. Subclasses must override the abstract :attr:`parameter_names` property and :meth:`__call__` method.

    """

    @doc
    @abstractmethod
    def __call__(
        self,
        regressors: pd.DataFrame,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Compute the additive regressor contribution.

        Parameters
        ----------
        %(regressors)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
