"""Convolved regressor model."""

from keras import ops
from prfmodel._docstring import doc
from prfmodel.impulse._convolve import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.typing import Tensor
from prfmodel.utils import ParamsDict
from .base import BaseRegressors


class ConvolvedRegressors(BaseRegressors):
    r"""
    Convolved regressor model.

    Adds a linear combination of regressors after convolving each with an impulse response. Suitable for task or
    event regressors that are specified at the neural-signal level and need to be transformed into the predicted
    response space (e.g., BOLD space) via the impulse model.

    Given a design matrix with columns :math:`x_k(t)`, an impulse response :math:`h(t)`, and per-unit beta weights
    :math:`\beta_k`, the model prediction is:

    .. math::

        y(t) = \sum_k \beta_k \, (x_k * h)(t).

    The impulse response is generated per unit from the same `parameters` data frame, so impulse parameters can be
    shared with (or differ from) those of a parent canonical model. The design is supplied at call time as a
    :class:`pandas.DataFrame`. The model selects the columns it needs by name, so column order is unimportant and
    extra columns are silently ignored.

    Parameters
    ----------
    names : list of str
        Names of the regressors. Each name must appear as a column in the ``regressors`` data frame supplied to the
        :meth:`__call__` method. Parameter names are derived as ``"beta_<name>"`` for each name.
    impulse_model : BaseImpulse
        The impulse model used to convolve each regressor with. Typically the same instance used by the parent
        canonical model, so that impulse parameter names dedupe naturally during parameter aggregation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.impulse import DerivativeTwoGammaImpulse
    >>> hrf = DerivativeTwoGammaImpulse()
    >>> regressors_model = ConvolvedRegressors(names=["task"], impulse_model=hrf)
    >>> sorted(set(regressors_model.parameter_names) - set(hrf.parameter_names))
    ['beta_task']
    >>> num_frames = 20
    >>> design = pd.DataFrame({"task": np.zeros(num_frames)})
    >>> design.loc[5, "task"] = 1.0
    >>> params = pd.DataFrame({
    ...     "beta_task": [1.0, -1.0],
    ...     # delay, dispersion, undershoot, u_dispersion, and ratio use the default Glover HRF parameters
    ...     "weight_deriv": [0.5, 0.5],
    ... })
    >>> resp = regressors_model(design, params)
    >>> print(resp.shape)
    (2, 20)

    """

    def __init__(self, names: list[str], impulse_model: BaseImpulse):
        super().__init__()

        self.names = list(names)
        self.impulse_model = impulse_model

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Includes ``beta_<name>`` for each regressor name as well as the parameter names of the impulse model.

        """
        beta_names = [f"beta_{name}" for name in self.names]
        return list(dict.fromkeys(beta_names + self.impulse_model.parameter_names))

    @doc
    def call(self, regressors: ParamsDict, parameters: ParamsDict) -> Tensor:
        """
        Compute the convolved regressor prediction.

        Parameters
        ----------
        %(regressors_tensors)s
        %(parameters_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        beta_names = [f"beta_{name}" for name in self.names]
        dtype = parameters.dtype

        design = regressors[self.names]

        num_units = parameters.shape[0]
        num_frames = design.shape[0]

        impulse = self.impulse_model.call(parameters)

        betas = parameters[beta_names]

        prediction = ops.zeros((num_units, num_frames), dtype=dtype)

        for idx in range(design.shape[1]):
            reg = ops.broadcast_to(design[:, idx], (num_units, num_frames))
            convolved = convolve_prf_impulse_response(reg, impulse, dtype=dtype)
            prediction = prediction + convolved * ops.expand_dims(betas[:, idx], axis=-1)

        return prediction
