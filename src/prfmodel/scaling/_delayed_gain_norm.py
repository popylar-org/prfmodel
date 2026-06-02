"""Delayed gain normalization scaling model."""

import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.exceptions import ShapeError
from prfmodel.impulse._convolve import convolve_prf_impulse_response
from prfmodel.typing import Tensor
from prfmodel.utils import _EXPECTED_NDIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseScaling


class DelayedGainNormScaling(BaseScaling):
    r"""
    Delayed gain normalization scaling model.

    Applies a temporal gain control formula to the impulse-convolved encoded response L(t),
    the normalization signal is an exponential low-pass filtered version of L(t) itself.

    Parameters
    ----------
    duration : float, default=32.0
        Duration of the exponential decay kernel in seconds. Same convention as
        :attr:`~prfmodel.impulse.base.BaseImpulse.duration`.
    resolution : float, default=1.0
        Seconds per frame. Same convention as
        :attr:`~prfmodel.impulse.base.BaseImpulse.resolution`.

    Notes
    -----
    Given the impulse-convolved encoded response :math:`L(t)` with shape
    ``(num_units, num_frames)``, the output is:

    .. math::

        \text{output}(t) = \text{amplitude} \cdot R(t) + \text{baseline}

    where

    .. math::

        R(t) = \frac{|L(t)|^n}{\sigma^n + |(L * h_2)(t)|^n}

    and :math:`h_2(t) = \exp(-t / \tau_2)` is a causal exponential decay kernel
    parameterized by :math:`\tau_2`, normalised to sum to one.

    References
    ----------
    .. [1] Zhou J., Benson N.C., Kay K., Winawer J. (2019). Predicting neuronal dynamics with a
        delayed gain control model. *PLOS Computational Biology*, 15(9).
        https://doi.org/10.1371/journal.pcbi.1007484

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> params = pd.DataFrame({
    ...     "n": [2.0, 1.5],
    ...     "tau_2": [0.1, 0.2],
    ...     "sigma_saturation": [1.0, 2.0],
    ...     "amplitude": [1.0, 2.0],
    ...     "baseline": [0.0, 0.5],
    ... })
    >>> inputs = np.ones((2, 20))
    >>> model = DelayedGainNormScaling()
    >>> resp = model(inputs, params)
    >>> print(resp.shape)
    (2, 20)

    """

    def __init__(self, duration: float = 32.0, resolution: float = 1.0):
        self.duration = duration
        self.resolution = resolution
        self._t_cached: Tensor | None = None

    @property
    def _frames(self) -> Tensor:
        """Cached time axis for the exponential kernel, shape (1, num_kernel_frames)."""
        if self._t_cached is None:
            num_kernel_frames = int(self.duration / self.resolution)
            self._t_cached = ops.expand_dims(
                ops.linspace(0.0, self.duration, num_kernel_frames),
                0,
            )
        return self._t_cached

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Parameter names are: ``n``, ``tau_2``, ``sigma_saturation``, ``amplitude``, ``baseline``.

        """
        return ["n", "tau_2", "sigma_saturation", "amplitude", "baseline"]

    @doc
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the delayed gain normalization model response.

        Parameters
        ----------
        inputs : :data:`prfmodel.typing.Tensor`
            Impulse-convolved encoded response L(t) with shape ``(num_units, num_frames)``.
        %(parameters)s

            - ``n`` : Exponent for the nonlinear stage. Must be >= 1 for all units.
              When ``n < 1``, the exponentiation compresses the response and the formula
              no longer behaves as an expansive gain control.
            - ``tau_2`` : Time constant of the exponential decay kernel h₂ in seconds.
              When ``tau_2 <= 0``, the kernel is undefined or explosive.
            - ``sigma_saturation`` : Semi-saturation constant. When ``sigma_saturation <= 0``
              and the gain signal is also zero (silent stimulus), the denominator is zero and
              the response is undefined.
            - ``amplitude`` : Multiplicative output scale.
            - ``baseline`` : Additive output constant.
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
        dtype = get_dtype(dtype)
        inputs = ops.convert_to_tensor(inputs, dtype=dtype)

        if len(inputs.shape) != _EXPECTED_NDIM:
            raise ShapeError("inputs", inputs.shape, f"must have exactly {_EXPECTED_NDIM} dimensions")  # noqa: EM101

        n_values = parameters["n"].to_numpy()
        if (n_values < 1.0).any():
            bad = n_values[n_values < 1.0].tolist()
            msg = f"All values of 'n' must be >= 1, but got: {bad}"
            raise ValueError(msg)

        n = convert_parameters_to_tensor(parameters[["n"]], dtype=dtype)
        tau_2 = convert_parameters_to_tensor(parameters[["tau_2"]], dtype=dtype)
        sigma_saturation = convert_parameters_to_tensor(parameters[["sigma_saturation"]], dtype=dtype)
        amplitude = convert_parameters_to_tensor(parameters[["amplitude"]], dtype=dtype)
        baseline = convert_parameters_to_tensor(parameters[["baseline"]], dtype=dtype)

        # Numerator: |L(t)|^n
        r_ln = ops.power(ops.abs(inputs), n)

        # Build per-unit exponential decay kernel h2: exp(-t / tau_2)
        # t: (1, num_kernel_frames), tau_2: (num_units, 1) -> kernel: (num_units, num_kernel_frames)
        t = ops.cast(self._frames, dtype=dtype)
        kernel = ops.exp(-t / tau_2)
        kernel = kernel / ops.sum(kernel, axis=1, keepdims=True)  # normalise by sum

        # Gain signal: convolve L(t) with h2 (abs applied after convolution)
        g_t = convolve_prf_impulse_response(inputs, kernel)

        # Denominator: sigma^n + |G(t)|^n
        denominator = ops.power(sigma_saturation, n) + ops.power(ops.abs(g_t), n)

        r_t = r_ln / denominator

        return amplitude * r_t + baseline
