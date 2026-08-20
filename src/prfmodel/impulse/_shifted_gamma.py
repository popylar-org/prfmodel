"""Shifted gamma distribution impulse response."""

from typing import ClassVar
from prfmodel._docstring import doc
from prfmodel.density._gamma import shifted_gamma_density
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import normalize_response
from .base import BaseImpulse


class ShiftedGammaImpulse(BaseImpulse):
    r"""
    Shifted gamma distribution impulse model.

    Predicts an impulse response that is a shifted gamma distribution.
    The model has three parameters: `delay` is the mean time (in seconds) of the gamma distribution, `dispersion`
    its scale, and `shift` its onset.

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
    default_parameters : dict of float or str, optional
        Dictionary with scalar default parameter values or name of default parameter set.
        Dictionary keys must be valid parameter names. Default values can be overriden in the :meth:`__call__` method.

    See Also
    --------
    gamma_density : Density of the gamma distribution.
    shifted_gamma_density : Shifted density of the gamma distribution.

    Notes
    -----
    The predicted impulse response at time :math:`t` with :math:`\alpha = delay / dispersion`,
    :math:`\theta = dispersion`, and :math:`\delta = shift` is:

    .. math::

        f(t) = f_{\text{gamma}}(t - \delta; \alpha, \theta)

    The response prior to the onset of the gamma distribution is set to zero.

    `dispersion` is the gamma **scale**, following the convention used by SPM, nilearn and Glover.
    Consequently the mean of the gamma distribution is exactly `delay` seconds after `shift`.

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
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "delay": [2.0, 1.0, 1.5],
    ...     "dispersion": [1.0, 1.0, 1.0],
    ...     "shift": [1.0, 2.0, 5.0],
    ... })
    >>> impulse_model = ShiftedGammaImpulse()
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 32)

    """

    _positive_parameter_names: ClassVar[tuple[str, ...]] = ("delay", "dispersion")

    @property
    def _all_parameter_names(self) -> list[str]:
        """Parameter names are: `delay`, `dispersion`, and `shift`."""
        return ["delay", "dispersion", "shift"]

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
        shift = parameters[["shift"]]

        dens = shifted_gamma_density(frames, delay / dispersion, dispersion, shift)

        return normalize_response(dens, self.norm)
