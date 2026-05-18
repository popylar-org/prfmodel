"""Weighted difference of two derivative gamma distribution impulse response."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.density._gamma import derivative_gamma_density
from prfmodel.density._gamma import gamma_density
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from prfmodel.utils import normalize_response
from .base import BaseImpulse
from .defaults import _fetch_default


class DerivativeTwoGammaImpulse(BaseImpulse):
    r"""
    Weighted difference of two derivative gamma distributions impulse model.

    Predicts an impulse response that is the weighted derivative difference of two gamma distributions. This
    weighted derivative difference is added to the weighted difference of the two gamma distributions.
    The model has six parameters: `delay` and `undershoot` refer to the positive and negative peaks of the response
    while `dispersion` and `u_dispersion` refer to the rate parameters of the two gamma distributions. The `ratio`
    parameter indicates the weight of the second gamma distribution. The `weight_deriv` represents the weight of the
    derivative difference added to the standard difference.

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
        The normalization of the response. Can be `"sum"` (default), `"mean"`, `"max"`, `"norm"`, or `None`. If `None`,
        no normalization is performed.
    default_parameters : dict of float or str, optional, default="glover_hrf"
        Dictionary with scalar default parameter values or name of default parameter set. Available default
        parameter sets are `glover_hrf` (default) and `spm_hrf`. See :mod:`~prfmodel.impulse.defaults` for details.
        Dictionary keys must be valid parameter names. Default values can be overriden in the :meth:`__call__` method.

    See Also
    --------
    TwoGammaImpulse : Weighted difference of two gamma distributions impulse model.
    gamma_density : Density of the gamma distribution.
    derivative_gamma_density : Derivative density of the gamma distribution.

    Notes
    -----
    The predicted impulse response at time :math:`t` with :math:`\alpha_1 = delay / dispersion`,
    :math:`\lambda_1 = dispersion`, :math:`\alpha_2  = undershoot / u\_dispersion`, :math:`\lambda_2 = u\_dispersion`,
    :math:`\omega = ratio`, and :math:`\tau = weight\_deriv` is:

    .. math::

        f(t) = f_{\text{diff}}(t) - \tau f'_{\text{diff}}(t)

    .. math::

        f_{\text{diff}}(t) = f_{\text{gamma}}(t; \alpha_1, \lambda_1) - \omega
            f_{\text{gamma}}(t; \alpha_2, \lambda_2)

    Positive `weight_deriv` values shift the response to the right.

    References
    ----------
    .. [1] Boynton, G. M., Engel, S. A., Glover, G. H., & Heeger, D. J. (1996). Linear systems analysis of functional
        magnetic resonance imaging in human V1. *The Journal of Neuroscience*, 16(13), 4207-4221.
        https://doi.org/10.1523/JNEUROSCI.16-13-04207.1996
    .. [2] Friston, K. J., Fletcher, P., Josephs, O., Holmes, A., Rugg, M. D., & Turner, R. (1998). Event-related fMRI:
        Characterizing differential responses. *NeuroImage*, 7(1), 30-40. https://doi.org/10.1006/nimg.1997.0306
    .. [3] Glover, G. H. (1999). Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage*, 9(4),
        416-429. https://doi.org/10.1006/nimg.1998.0419

    Examples
    --------
    Predict an impulse response using the default parameter set
    (:func:`~prfmodel.impulse.defaults.default_two_gamma_impulse_glover_hrf()`).

    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "weight_deriv": [0.5, -0.7, 0.9],
    ... })
    >>> impulse_model = DerivativeTwoGammaImpulse()
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 32)

    Predict an impulse response by overriding the default parameter set in the :meth:`__call__` method.

    >>> params = pd.DataFrame({
    ...     "delay": [2.0, 1.0, 1.5],
    ...     "dispersion": [1.0, 1.0, 1.0],
    ...     "undershoot": [1.5, 2.0, 1.0],
    ...     "u_dispersion": [1.0, 1.0, 1.0],
    ...     "ratio": [0.7, 0.2, 0.5],
    ...     "weight_deriv": [0.5, -0.7, 0.9],
    ... })
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 32)

    If ``default_parameters=None``, all parameters must be supplied to :meth:`__call__`.

    >>> impulse_model = DerivativeTwoGammaImpulse(
    ...     default_parameters=None,
    ... )
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 32)

    """

    def __init__(
        self,
        duration: float = 32.0,
        offset: float = 0.0001,
        resolution: float = 1.0,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | str | None = "glover_hrf",
    ):
        if isinstance(default_parameters, str):
            default_parameters = _fetch_default(default_parameters)

        super().__init__(duration, offset, resolution, norm, default_parameters)

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`, `weight_deriv`.

        """
        return ["delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv"]

    @doc
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        %(parameters)s Parameter values override default parameters.
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The predicted impulse response with shape `(num_units, num_frames)` and dtype `dtype`.

        """
        parameters = self._join_default_parameters(parameters)
        dtype = get_dtype(dtype)
        frames = ops.cast(self.frames, dtype=dtype)

        delay = convert_parameters_to_tensor(parameters[["delay"]], dtype=dtype)
        dispersion = convert_parameters_to_tensor(parameters[["dispersion"]], dtype=dtype)
        undershoot = convert_parameters_to_tensor(parameters[["undershoot"]], dtype=dtype)
        u_dispersion = convert_parameters_to_tensor(parameters[["u_dispersion"]], dtype=dtype)
        ratio = convert_parameters_to_tensor(parameters[["ratio"]], dtype=dtype)
        weight_deriv = convert_parameters_to_tensor(parameters[["weight_deriv"]], dtype=dtype)

        dens_1 = gamma_density(frames, delay / dispersion, dispersion)
        dens_2 = gamma_density(frames, undershoot / u_dispersion, u_dispersion)

        dens_deriv_1 = derivative_gamma_density(frames, delay / dispersion, dispersion)
        dens_deriv_2 = derivative_gamma_density(frames, undershoot / u_dispersion, u_dispersion)

        diff_dens = dens_1 - ratio * dens_2
        diff_dens_deriv = dens_deriv_1 - ratio * dens_deriv_2

        return normalize_response(diff_dens - weight_deriv * diff_dens_deriv, self.norm)
