"""Weighted difference of two gamma distribution impulse response."""

import pandas as pd
from keras import ops
from prfmodel.models.base import BaseImpulse
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .density import gamma_density


class TwoGammaImpulse(BaseImpulse):
    r"""
    Weighted difference of two gamma distributions impulse response model.

    Predicts an impulse response that is the weighted difference of two gamma distributions.
    The model has five parameters: `shape_1` and `rate_1` for the first, `shape_2` and `rate_2` for the second
    gamma distribution, and `weight` for the relative weight of the first gamma distribution.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0001
        The offset of the impulse response (in seconds). By default a very small offset is added to prevent infinite
        response values at t = 0.
    resolution : float, default=1.0
        The time resultion of the impulse response (in seconds), that is the number of points per second at which the
        impulse response function is evaluated.
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Keys must be valid parameter names.

    Notes
    -----
    The predicted impulse response at time :math:`t` with `shape_1` :math:`\alpha_1`, `rate_1` :math:`\lambda_1`,
    `shape_2` :math:`\alpha_2`, `rate_2` :math:`\lambda_2`, and `weight` :math:`w` is:

    .. math::

        f(t) = \hat{f}_{\text{gamma}}(t; \alpha_1, \lambda_1) - w \hat{f}_{\text{gamma}}(t; \alpha_2, \lambda_2)

    The gamma distributions are divided by their respective maximum, so that their highest peak has an amplitude of 1:

    .. math::
        \hat{f}_{\text{gamma}}(t; \alpha, \lambda) = \frac{f_{\text{gamma}}(t; \alpha, \lambda)}
        {\text{max}(f_{\text{gamma}}(t; \alpha, \lambda))}

    Examples
    --------
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "shape_1": [2.0, 1.0, 1.5],
    >>>     "rate_1": [1.0, 1.0, 1.0],
    >>>     "shape_2": [1.5, 2.0, 1.0],
    >>>     "rate_2": [1.0, 1.0, 1.0],
    >>>     "weight": [0.7, 0.2, 0.5],
    >>> })
    >>> impulse_model = TwoGammaImpulse(
    >>>     duration=100.0 # 100 seconds
    >>> )
    >>> resp = impulse_model(params)
    >>> print(resp.shape) # (num_rows, duration)
    (3, 100)

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `shape_1`, `rate_1`, `shape_2`, `rate_2`, `weight`.

        """
        return ["shape_1", "rate_1", "shape_2", "rate_2", "weight"]

    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches. Must contain the columns `shape_1`, `rate_1`, `shape_2`, `rate_2`, and `weight`.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            The predicted impulse response with shape `(num_batches, num_frames)` and dtype `dtype`.

        """
        parameters = self._join_default_parameters(parameters)
        dtype = get_dtype(dtype)
        frames = ops.cast(self.frames, dtype=dtype)
        shape_1 = convert_parameters_to_tensor(parameters[["shape_1"]], dtype=dtype)
        rate_1 = convert_parameters_to_tensor(parameters[["rate_1"]], dtype=dtype)
        shape_2 = convert_parameters_to_tensor(parameters[["shape_2"]], dtype=dtype)
        rate_2 = convert_parameters_to_tensor(parameters[["rate_2"]], dtype=dtype)
        weight = convert_parameters_to_tensor(parameters[["weight"]], dtype=dtype)
        # Compute unnormalized density because normalizing constant cancels out when taking difference anyway
        dens_1 = gamma_density(frames, shape_1, rate_1, norm=False)
        dens_1_norm = dens_1 / ops.max(dens_1, axis=1, keepdims=True)
        dens_2 = gamma_density(frames, shape_2, rate_2, norm=False)
        dens_2_norm = dens_2 / ops.max(dens_2, axis=1, keepdims=True)
        return dens_1_norm - weight * dens_2_norm
