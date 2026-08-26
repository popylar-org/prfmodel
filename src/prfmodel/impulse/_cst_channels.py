"""Temporal channel impulse responses of the compressive spatiotemporal (CST) model."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.density._gamma import gamma_density
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from prfmodel.utils import normalize_response
from .base import BaseImpulse


class SustainedImpulse(BaseImpulse):
    r"""
    Sustained temporal channel impulse model.

    Predicts the sustained impulse response :math:`h_1` of the compressive spatiotemporal (CST) model [1]_.
    The model has one parameter: `time_to_peak` is the time (in seconds) at which the response peaks.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0
        The offset of the impulse response (in seconds).
    resolution : float, default=1.0
        The time resolution of the impulse response (in seconds), that is the number of points per second at which
        the impulse response function is evaluated.
    norm : str, optional, default=None
        The normalization of the response. Can be `"sum"`, `"mean"`, `"max"`, `"norm"`, or `None` (default).
        If `None`, no normalization is performed. The default is `None` because the gamma density already carries
        its own normalizing constant, as in the reference implementation.
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Dictionary keys must be valid parameter names.
        Default values can be overriden in the :meth:`__call__` method.
    shape : float, default=9.0
        Shape of the gamma distribution (:math:`m` in the reference). Fixed at 9 in [1]_.

    See Also
    --------
    gamma_density : Density of the gamma distribution.
    TransientImpulse : The transient temporal channel of the same model.

    Notes
    -----
    The reference [1]_ writes the impulse response with a time constant :math:`\tau` and shape :math:`m` as:

    .. math::

        h(t) = \frac{(t / \tau)^{m - 1} e^{-t / \tau}}{\tau (m - 1)!}

    which is the density of a gamma distribution with shape :math:`m` and scale :math:`\tau`. That density peaks at
    :math:`(m - 1)\tau`, so this class takes the peak time itself as its parameter and solves for the scale:

    .. math::

        \tau = \frac{\mathtt{time\_to\_peak}}{m - 1}

    This makes `time_to_peak` literally the time of the maximum, which is the quantity [1]_ reports across visual
    areas (roughly 50 ms in V1 rising to 230 ms in IPS).

    References
    ----------
    .. [1] Kim, I., Kupers, E. R., Lerma-Usabiaga, G., & Grill-Spector, K. (2024). Characterizing spatiotemporal
        population receptive fields in human visual cortex with fMRI. *The Journal of Neuroscience*, 44(2),
        e0803232023. https://doi.org/10.1523/JNEUROSCI.0803-23.2023

    Examples
    --------
    >>> import pandas as pd
    >>> impulse_model = SustainedImpulse(duration=2.0, resolution=0.01)
    >>> params = pd.DataFrame({"time_to_peak": [0.05, 0.1, 0.2]})
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 200)

    """

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        duration: float = 32.0,
        offset: float = 0.0,
        resolution: float = 1.0,
        norm: str | None = None,
        default_parameters: dict[str, float] | None = None,
        shape: float = 9.0,
    ):
        self.shape = shape

        super().__init__(duration, offset, resolution, norm, default_parameters)

    @property
    def _all_parameter_names(self) -> list[str]:
        """Parameter names are: `time_to_peak`."""
        return ["time_to_peak"]

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

        Raises
        ------
        %(raises_missing_parameters)s

        """
        parameters = self._join_default_parameters(parameters)
        dtype = get_dtype(dtype)
        frames = self.get_frames(dtype)
        time_to_peak = convert_parameters_to_tensor(parameters[["time_to_peak"]], dtype=dtype)

        dens = _gamma_at_scale(frames, _reference_scale(time_to_peak, self.shape), self.shape)

        return normalize_response(dens, self.norm)


class TransientImpulse(BaseImpulse):
    r"""
    Transient temporal channel impulse model.

    Predicts the on-transient impulse response :math:`h_2` of the compressive spatiotemporal (CST) model [1]_, a
    biphasic response formed as the difference between an excitatory and a slower inhibitory gamma density.
    The model has one parameter: `time_to_peak` is the peak time (in seconds) of the excitatory component.

    The off-transient :math:`h_3` is identical with the opposite sign, so it is obtained by negating this response
    rather than by a separate class.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0
        The offset of the impulse response (in seconds).
    resolution : float, default=1.0
        The time resolution of the impulse response (in seconds), that is the number of points per second at which
        the impulse response function is evaluated.
    norm : str, optional, default=None
        The normalization of the response. Can be `"sum"`, `"mean"`, `"max"`, `"norm"`, or `None` (default).
        If `None`, no normalization is performed. Leave this at `None`: the two components each integrate to one, so
        a biphasic response sums to approximately zero and `norm="sum"` divides by that near-zero value.
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Dictionary keys must be valid parameter names.
        Default values can be overriden in the :meth:`__call__` method.
    shape : float, default=9.0
        Shape of the excitatory gamma distribution (:math:`m` in the reference). Fixed at 9 in [1]_.
    inhibitory_shape : float, default=10.0
        Shape of the inhibitory gamma distribution. Fixed at 10 in [1]_.
    inhibitory_time_constant_ratio : float, default=1.33
        Scale of the inhibitory gamma distribution relative to the excitatory one (:math:`\kappa` in the
        reference). Fixed at 1.33 in [1]_. Values above one make the inhibitory component slower, which is what
        produces the biphasic shape.

    See Also
    --------
    SustainedImpulse : The sustained temporal channel of the same model.

    Notes
    -----
    The excitatory component is identical to :class:`SustainedImpulse`, so it peaks at `time_to_peak`. The
    inhibitory component shares its scale but is stretched by `inhibitory_time_constant_ratio`, so it peaks at

    .. math::

        \frac{(m_{inh} - 1) \kappa}{m - 1} \times \mathtt{time\_to\_peak} \approx 1.496 \times
        \mathtt{time\_to\_peak}

    Because the inhibitory component peaks later, the difference peaks *earlier* than the sustained channel and
    then crosses zero once into a negative lobe.

    References
    ----------
    .. [1] Kim, I., Kupers, E. R., Lerma-Usabiaga, G., & Grill-Spector, K. (2024). Characterizing spatiotemporal
        population receptive fields in human visual cortex with fMRI. *The Journal of Neuroscience*, 44(2),
        e0803232023. https://doi.org/10.1523/JNEUROSCI.0803-23.2023

    Examples
    --------
    >>> import pandas as pd
    >>> impulse_model = TransientImpulse(duration=2.0, resolution=0.01)
    >>> params = pd.DataFrame({"time_to_peak": [0.05, 0.1, 0.2]})
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 200)

    """

    def __init__(  # noqa: PLR0913 (too many arguments)
        self,
        duration: float = 32.0,
        offset: float = 0.0,
        resolution: float = 1.0,
        norm: str | None = None,
        default_parameters: dict[str, float] | None = None,
        shape: float = 9.0,
        inhibitory_shape: float = 10.0,
        inhibitory_time_constant_ratio: float = 1.33,
    ):
        self.shape = shape
        self.inhibitory_shape = inhibitory_shape
        self.inhibitory_time_constant_ratio = inhibitory_time_constant_ratio

        super().__init__(duration, offset, resolution, norm, default_parameters)

    @property
    def _all_parameter_names(self) -> list[str]:
        """Parameter names are: `time_to_peak`."""
        return ["time_to_peak"]

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

        Raises
        ------
        %(raises_missing_parameters)s

        """
        parameters = self._join_default_parameters(parameters)
        dtype = get_dtype(dtype)
        frames = self.get_frames(dtype)
        time_to_peak = convert_parameters_to_tensor(parameters[["time_to_peak"]], dtype=dtype)

        scale = _reference_scale(time_to_peak, self.shape)

        dens_excitatory = _gamma_at_scale(frames, scale, self.shape)
        dens_inhibitory = _gamma_at_scale(frames, self.inhibitory_time_constant_ratio * scale, self.inhibitory_shape)

        return normalize_response(dens_excitatory - dens_inhibitory, self.norm)


def _reference_scale(time_to_peak: Tensor, shape: float) -> Tensor:
    """Return the gamma scale that places the peak of a shape-`shape` gamma density at `time_to_peak`.

    The gamma density with shape `m` and scale `s` peaks at `(m - 1) * s`, so the scale that puts the peak at
    `time_to_peak` is `time_to_peak / (m - 1)`.

    """
    return time_to_peak / (shape - 1)


def _gamma_at_scale(frames: Tensor, scale: Tensor, shape: float) -> Tensor:
    """Evaluate a gamma density of shape `shape` at `scale`.

    The shape is broadcast against `scale` because :func:`~prfmodel.density.gamma_density` requires its `shape` and
    `scale` arguments to have matching shapes.

    """
    return gamma_density(frames, ops.ones_like(scale) * shape, scale)
