"""Composite regressor model that aggregates multiple regressor models."""

import pandas as pd
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from .base import BaseRegressors


class RegressorsList(BaseRegressors):
    """
    Composite regressor model that sums the predictions of multiple regressor models.

    Used internally by canonical models to support passing a list of regressor models as the ``regressors_model``
    argument. The parameter names of all child regressor models are aggregated (preserving insertion order, removing
    duplicates).

    At call time, the supplied design is a single :class:`pandas.DataFrame` that is passed to every child, each of
    which slices the columns it needs by name.

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

        self.regressors = list(regressors)

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model, aggregated from all child regressor models."""
        names: list[str] = []

        for regressor in self.regressors:
            names.extend(regressor.parameter_names)

        return list(dict.fromkeys(names))

    @doc
    def __call__(
        self,
        regressors: pd.DataFrame,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Compute the sum of predictions from all child regressor models.

        Parameters
        ----------
        regressors : pandas.DataFrame
            A single design data frame whose columns cover every child's required regressor names. It is passed to
            each child, which slices the columns it needs by name; extra columns are ignored.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        prediction = self.regressors[0](regressors, parameters, dtype=dtype)

        for child in self.regressors[1:]:
            prediction = prediction + child(regressors, parameters, dtype=dtype)

        return prediction
