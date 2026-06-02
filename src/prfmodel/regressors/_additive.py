"""Additive regressor model."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseRegressors
from .base import _extract_design


class AdditiveRegressors(BaseRegressors):
    r"""
    Additive regressor model.

    Adds a linear combination of regressors directly to the model prediction without further transformation. Suitable
    for nuisance regressors such as motion parameters, drift terms, or physiological signals that are already in the
    space of the predicted response.

    Given a design matrix with columns :math:`x_k(t)` and per-unit beta weights :math:`\beta_k`, the model prediction
    is:

    .. math::

        y(t) = \sum_k \beta_k \, x_k(t).

    The design is supplied at call time as a :class:`pandas.DataFrame`. The model selects the columns it needs by
    name, so column order is unimportant and extra columns are silently ignored.

    Parameters
    ----------
    names : list of str
        Names of the regressors. Each name must appear as a column in the ``regressors`` data frame supplied to the
        :meth:`__call__` method. Parameter names are derived as ``"beta_<name>"`` for each name.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> regressors_model = AdditiveRegressors(names=["motion_x", "motion_y"])
    >>> regressors_model.parameter_names
    ['beta_motion_x', 'beta_motion_y']
    >>> design = pd.DataFrame({"motion_x": np.ones(10), "motion_y": np.ones(10)})
    >>> params = pd.DataFrame({"beta_motion_x": [1.0, 2.0], "beta_motion_y": [-1.0, 0.5]})
    >>> resp = regressors_model(design, params)
    >>> print(resp.shape)
    (2, 10)

    """

    def __init__(self, names: list[str]):
        super().__init__()

        self.names = list(names)

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: ``beta_<name>`` for each regressor name."""
        return [f"beta_{name}" for name in self.names]

    @doc
    def __call__(
        self,
        regressors: pd.DataFrame,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Compute the additive regressor prediction.

        Parameters
        ----------
        %(regressors)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)

        design_df = _extract_design(regressors, self.names)
        design = ops.convert_to_tensor(design_df.to_numpy(), dtype=dtype)
        betas = convert_parameters_to_tensor(parameters[self.parameter_names], dtype=dtype)

        return ops.matmul(betas, ops.transpose(design))
