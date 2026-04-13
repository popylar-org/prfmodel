"""Shifted gamma distribution impulse response."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from prfmodel.utils import normalize_response
from .base import BaseImpulse
from .density import shifted_gamma_density


class ShiftedGammaImpulse(BaseImpulse):
    r"""
    Shifted gamma distribution impulse response model.

    Predicts an impulse response that is a shifted gamma distribution.
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
        The normalization of the response. Can be `"sum"` (default), `"mean"`, `"max"`, `"norm"`, or `None`. If `None`,
        no normalization is performed.
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Keys must be valid parameter names.

    See Also
    --------
    gamma_density : Density of the gamma distribution.
    shifted_gamma_density : Shifted density of the gamma distribution.

    Notes
    -----
    The predicted impulse response at time :math:`t` with :math:`\alpha = delay / dispersion`,
    :math:`\lambda = dispersion`, and :math:`\delta = shift` is:

    .. math::

        f(t) = f_{\text{gamma}}(t - \delta; \alpha, \lambda)

    The response prior to the onset of the gamma distribution is set to zero.

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
    >>> impulse_model = ShiftedGammaImpulse(
    ...     duration=100.0  # 100 seconds
    ... )
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 100)

    """

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: `delay`, `dispersion`, and `shift`.

        """
        return ["delay", "dispersion", "shift"]

    @doc
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        %(parameters)s
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
        shift = convert_parameters_to_tensor(parameters[["shift"]], dtype=dtype)

        dens = shifted_gamma_density(frames, delay / dispersion, dispersion, shift)

        return normalize_response(dens, self.norm)
