"""Weighted difference of two gamma distribution impulse response."""

import pandas as pd
from keras import ops
from prfmodel.models.base import BaseImpulse
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from prfmodel.utils import normalize_response
from .density import gamma_density


class TwoGammaImpulse(BaseImpulse):
    r"""
    Weighted difference of two gamma distributions impulse response model.

    Predicts an impulse response that is the weighted difference of two gamma distributions.
    The model has five parameters: `delay` and `undershoot` refer to the positive and negative peaks of the response
    while `dispersion` and `u_dispersion` refer to the rate parameters of the two gamma distributions. The `ratio`
    parameter indicates the weight of the second gamma distribution.

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
    The predicted impulse response at time :math:`t` with :math:`\alpha_1 = delay / dispersion`,
    :math:`\lambda_1 = dispersion`, :math:`\alpha_2  = undershoot / u\_dispersion`, :math:`\lambda_2 = u\_dispersion`,
    and :math:`\omega = ratio` is:

    .. math::

        f(t) = f_{\text{gamma}}(t; \alpha_1, \lambda_1) - \omega f_{\text{gamma}}(t; \alpha_2, \lambda_2)

    See Also
    --------
    gamma_density : Density of the gamma distribution.

    Examples
    --------
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    >>>     "delay": [2.0, 1.0, 1.5],
    >>>     "dispersion": [1.0, 1.0, 1.0],
    >>>     "undershoot": [1.5, 2.0, 1.0],
    >>>     "u_dispersion": [1.0, 1.0, 1.0],
    >>>     "ratio": [0.7, 0.2, 0.5],
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

        Parameter names are: `delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`.

        """
        return ["delay", "dispersion", "undershoot", "u_dispersion", "ratio"]

    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches. Must contain the columns `delay`, `dispersion`, `undershoot`, `u_dispersion`,
            and `ratio`.
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
        undershoot = convert_parameters_to_tensor(parameters[["undershoot"]], dtype=dtype)
        u_dispersion = convert_parameters_to_tensor(parameters[["u_dispersion"]], dtype=dtype)
        ratio = convert_parameters_to_tensor(parameters[["ratio"]], dtype=dtype)

        dens_1 = gamma_density(frames, delay / dispersion, dispersion)
        dens_2 = gamma_density(frames, undershoot / u_dispersion, u_dispersion)

        return normalize_response(dens_1 - ratio * dens_2, self.norm)
