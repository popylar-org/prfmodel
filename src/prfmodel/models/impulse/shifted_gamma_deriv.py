"""Shifted derivative gamma distribution impulse response."""

import pandas as pd
from keras import ops
from prfmodel.models.base import BaseImpulse
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from prfmodel.utils import normalize_response
from .density import shifted_derivative_gamma_density


class ShiftedDerivativeGammaImpulse(BaseImpulse):
    r"""
    Shifted derivative of the gamma distribution impulse response model.

    Predicts an impulse response that is a shifted derivative of the gamma distribution.
    The model has three parameters: `delay` refers to the positive peak, `dispersion` to the
    rate, and `shift` to the onset of the gamma distribution.

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
    norm : str, optional, default="sum"
        The normalization of the response. Can be `"sum"` (default), `"mean"`, `"max"`, or `None`. If `None`, no
        normalization is performed.
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Keys must be valid parameter names.

    Notes
    -----
    The predicted impulse response at time math:`t` with :math:`\alpha = delay / dispersion`,
    :math:`\lambda = dispersion`, and :math:`\delta = shift` is:

    .. math::

        f(t) = f_{\text{gamma}}'(t - \delta; \alpha, \lambda)

    The response prior to the onset of the gamma distribution is set to zero.
    The derivative of the gamma distribution density with respect to time :math:`t` is calculated analytically
    as a function of the original gamma distribution density :math:`f_{\text{gamma}}(t; \alpha, \lambda)`:

    .. math::

        f_{\text{gamma}}'(t; \alpha, \lambda) = f_{\text{gamma}}(t; \alpha, \lambda)
        \frac{(\alpha - 1)}{t} - \lambda

    See Also
    --------
    shifted_gamma_density : Shifted density of the gamma distribution.
    shifted_derivative_gamma_density : Shifted derivative density of the gamma distribution.

    Examples
    --------
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "delay": [2.0, 1.0, 1.5],
    >>>     "dispersion": [1.0, 1.0, 1.0],
    >>>     "shift": [1.0, 2.0, 5.0],
    >>> })
    >>> impulse_model = ShiftedDerivativeGammaImpulse(
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

        Parameter names are: `delay`, `dispersion`, and `shift`.

        """
        return ["delay", "dispersion", "shift"]

    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches. Must contain the columns `delay`, `dispersion`, and `shift`.
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
        delay = convert_parameters_to_tensor(parameters[["delay"]], dtype=dtype)
        dispersion = convert_parameters_to_tensor(parameters[["dispersion"]], dtype=dtype)
        shift = convert_parameters_to_tensor(parameters[["shift"]], dtype=dtype)

        dens = shifted_derivative_gamma_density(frames, delay / dispersion, dispersion, shift)

        return normalize_response(dens)
