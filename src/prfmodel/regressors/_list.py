"""Composite regressor model that aggregates multiple regressor models."""

from collections.abc import Generator
from typing import cast
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from .base import BaseRegressors


class RegressorsList(BaseRegressors):
    """
    Composite regressor model that sums the predictions of multiple regressor models.

    Used internally by canonical models to support passing a list of regressor models as the ``regressors_model``
    argument. The parameter names of all child regressor models are aggregated (preserving insertion order, removing
    duplicates).

    At call time, the supplied design is a single :class:`pandas.DataFrame` or :class:`~prfmodel.utils.TensorFrame`
    that is passed to every child, each of which slices the columns it needs by name.

    Parameters
    ----------
    regressors : list of BaseRegressors
        Non-empty list of regressor model instances.

    Raises
    ------
    ValueError
        If `regressors` is empty.
    TypeError
        If any element is not a :class:`BaseRegressors` instance.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.regressors import AdditiveRegressors, RegressorsList
    >>> a = AdditiveRegressors(names=["x"])
    >>> b = AdditiveRegressors(names=["y"])
    >>> regressors_model = RegressorsList([a, b])
    >>> regressors_model.parameter_names
    ['beta_x', 'beta_y']
    >>> params = pd.DataFrame({"beta_x": [1.0], "beta_y": [1.0]})
    >>> design = pd.DataFrame({"x": np.ones(5), "y": np.ones(5) * 2.0})
    >>> resp = regressors_model(design, params)
    >>> print(resp.shape)
    (1, 5)

    """

    def __init__(self, regressors: list[BaseRegressors]):
        super().__init__()

        if not regressors:
            msg = "Argument 'regressors' must be a non-empty list of BaseRegressors instances"
            raise ValueError(msg)

        beta_names: list[str] = []

        for regressor in regressors:
            if not isinstance(regressor, BaseRegressors):
                msg = "All entries in 'regressors' must be instances of BaseRegressors"
                raise TypeError(msg)
            if any(name in beta_names for name in regressor.parameter_names):
                msg = "Regressor names must be unique"
                raise ValueError(msg)
            beta_names.extend(name for name in regressor.parameter_names if name.startswith("beta_"))

        self.models = {f"regressor_{i}": model for i, model in enumerate(regressors)}

    def _iter_models(self) -> Generator["BaseRegressors"]:
        return cast("Generator[BaseRegressors]", super()._iter_models())

    @property
    def regressor_names(self) -> list[str]:
        """Columns this model reads from the regressor design."""
        names: list[str] = []

        for model in self._iter_models():
            names.extend(model.regressor_names)

        return list(dict.fromkeys(names))

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
        for model in self._iter_models():
            model.check_regressor_names(regressors)

    @doc
    def call(self, regressors: TensorFrame, parameters: TensorFrame) -> Tensor:
        """
        Compute the sum of predictions from all child regressor models.

        Parameters
        ----------
        %(regressors_tensors)s
        %(parameters_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        first, *rest = self._iter_models()
        prediction = first.call(regressors, parameters)

        for model in rest:
            prediction = prediction + model.call(regressors, parameters)

        return prediction
