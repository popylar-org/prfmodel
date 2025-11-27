"""Shifted gamma distribution impulse response."""

import pandas as pd
from keras import ops
from prfmodel.models.base import BaseImpulse
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .density import shifted_gamma_density


class ShiftedGammaImpulse(BaseImpulse):
    r"""
    Shifted gamma distribution impulse response model.

    Predicts an impulse response that is a shifted gamma distribution.
    The model has three parameters: `shape`, `rate`, and `shift`.

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

    Notes
    -----
    The predicted impulse response at time :math:`t` with `shape` :math:`\alpha`, `rate` :math:`\lambda`,
    and `shift` :math:`\delta` is:

    .. math::

        f(t) = \hat{f}_{\text{gamma}}(t - \delta; \alpha, \lambda)

    The density of the gamma distribution is divided by its maximum,
    so that its highest peak has an amplitude of 1:

    .. math::
        \hat{f}_{\text{gamma}}(t; \alpha, \lambda, \delta) = \frac{f_{\text{gamma}}(t - \delta; \alpha, \lambda)}
        {\text{max}(f_{\text{gamma}}(t - \delta; \alpha, \lambda))}

    Examples
    --------
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "shape": [2.0, 1.0, 1.5],
    >>>     "rate": [1.0, 1.0, 1.0],
    >>>     "shift": [1.0, 2.0, 5.0],
    >>> })
    >>> impulse_model = ShiftedGammaImpulse(
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

        Parameter names are: `shape`, `rate`, and `shift`.

        """
        return ["shape", "rate", "shift"]

    def __call__(self, parameters: pd.DataFrame, dtype: str | None) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches. Must contain the columns `shape`, `rate`, and `shift`.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            The predicted impulse response with shape `(num_batches, num_frames)` and dtype `dtype`.

        """
        dtype = get_dtype(dtype)
        frames = ops.cast(self.frames, dtype=dtype)
        shape = convert_parameters_to_tensor(parameters[["shape"]], dtype=dtype)
        rate = convert_parameters_to_tensor(parameters[["rate"]], dtype=dtype)
        shift = convert_parameters_to_tensor(parameters[["shift"]], dtype=dtype)

        dens = shifted_gamma_density(frames, shape, rate, shift)

        return dens / ops.max(dens, axis=1, keepdims=True)
