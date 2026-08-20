"""Weighted difference of two derivative gamma distribution impulse response."""

from typing import ClassVar
from prfmodel._docstring import doc
from prfmodel.density._gamma import derivative_gamma_density
from prfmodel.density._gamma import gamma_density
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import normalize_response
from .base import BaseImpulse
from .defaults import _fetch_default


class DerivativeTwoGammaImpulse(BaseImpulse):
    r"""
    Weighted difference of two derivative gamma distributions impulse model.

    Predicts an impulse response that is the weighted derivative difference of two gamma distributions. This
    weighted derivative difference is added to the weighted difference of the two gamma distributions.
    The model has six parameters: `delay` and `undershoot` are the mean times (in seconds) of the positive and
    negative components of the response, while `dispersion` and `u_dispersion` are the scale parameters of the two
    gamma distributions. The `ratio` parameter indicates the weight of the second gamma distribution. The
    `weight_deriv` represents the weight of the derivative difference added to the standard difference.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0
        The offset of the impulse response (in seconds).
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
    :math:`\theta_1 = dispersion`, :math:`\alpha_2  = undershoot / u\_dispersion`,
    :math:`\theta_2 = u\_dispersion`, :math:`\omega = ratio`, and :math:`\tau = weight\_deriv` is:

    .. math::

        f(t) = f_{\text{diff}}(t) - \tau f'_{\text{diff}}(t)

    .. math::

        f_{\text{diff}}(t) = f_{\text{gamma}}(t; \alpha_1, \theta_1) - \omega
            f_{\text{gamma}}(t; \alpha_2, \theta_2)

    Positive `weight_deriv` values shift the response to the right.

    `dispersion` follows the convention used by SPM, nilearn and Glover, where it is the gamma **scale**.
    :func:`~prfmodel.density.gamma_density` is parameterized by the rate, so the reciprocal is taken here.
    Consequently the mean of the first component is exactly `delay` seconds and that of the second exactly
    `undershoot` seconds, and both parameters are directly comparable with values reported for other software.

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
        offset: float = 0.0,
        resolution: float = 1.0,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | str | None = "glover_hrf",
    ):
        if isinstance(default_parameters, str):
            default_parameters = _fetch_default(default_parameters)

        super().__init__(duration, offset, resolution, norm, default_parameters)

    _positive_parameter_names: ClassVar[tuple[str, ...]] = ("delay", "dispersion", "undershoot", "u_dispersion")

    @property
    def _all_parameter_names(self) -> list[str]:
        """Parameter names are: `delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`, `weight_deriv`."""
        return ["delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv"]

    @doc
    def call(self, parameters: TensorFrame) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        %(parameters_tensors)s :attr:`default_parameters` must already be merged in.


        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The predicted impulse response with shape `(num_units, num_frames)`.

        """
        parameters = self._join_default_parameters(parameters)
        dtype = parameters.dtype
        frames = self.get_frames(dtype)

        delay = parameters[["delay"]]
        dispersion = parameters[["dispersion"]]
        undershoot = parameters[["undershoot"]]
        u_dispersion = parameters[["u_dispersion"]]
        ratio = parameters[["ratio"]]
        weight_deriv = parameters[["weight_deriv"]]

        shape_scale_1 = (delay / dispersion, dispersion)
        shape_scale_2 = (undershoot / u_dispersion, u_dispersion)

        dens_1 = gamma_density(frames, *shape_scale_1)
        dens_2 = gamma_density(frames, *shape_scale_2)

        dens_deriv_1 = derivative_gamma_density(frames, *shape_scale_1)
        dens_deriv_2 = derivative_gamma_density(frames, *shape_scale_2)

        diff_dens = dens_1 - ratio * dens_2
        diff_dens_deriv = dens_deriv_1 - ratio * dens_deriv_2

        return normalize_response(diff_dens - weight_deriv * diff_dens_deriv, self.norm)
