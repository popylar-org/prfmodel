"""Impulse model base classes.

Classes in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example, :class:`~prfmodel.models.base.BaseImpulse` defines that all child classes
must implement a :meth:`~prfmodel.models.base.BaseImpulse.__call__` method that takes a set of parameters
as input. However, it leaves it up to each child class to define how input parameters are used to make
model predictions.

"""

from abc import abstractmethod
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol
from prfmodel.utils import _get_norm_fun
from prfmodel.utils import get_dtype


class BaseImpulse(ModelProtocol):
    """
    Abstract base class for impulse models.

    An impulse model takes a set of parameters as input a predicts an impulse for time frames that are
    defined by an offset, duration, and resolution.

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
        The normalization of the response. Can be `"sum"` (default), `"mean"`, `"max"`, `"norm"`, or `None`.
        If `None`, no normalization is performed.
    default_parameters : dict of float or str, optional
        Dictionary with scalar default parameter values or name of default parameter set.
        Dictionary keys must be valid parameter names. Default values can be overriden in the :meth:`__call__` method.

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom response
    models. Subclasses must override the abstract :attr:`_all_parameter_names` and :meth:`__call__` method.

    """

    def __init__(
        self,
        duration: float = 32.0,
        offset: float = 0.0001,
        resolution: float = 1.0,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | str | None = None,
    ):
        super().__init__()

        self.duration = duration
        self.offset = offset
        self.resolution = resolution

        # Check if norm arg is valid
        if norm is not None:
            _get_norm_fun(norm)

        self.norm = norm

        if isinstance(default_parameters, dict):
            if any(key not in self._all_parameter_names for key in default_parameters):
                msg = "Invalid default parameter name, please provide valid parameter default parameter names"
                raise ValueError(msg)

            if any(not isinstance(val, float) for val in default_parameters.values()):
                msg = "Default parameters must be single float values"
                raise ValueError(msg)

        self.default_parameters = default_parameters

        self._frames: dict[str, Tensor] = {}

    @property
    def num_frames(self) -> int:
        """The total number of time frames at which the impulse response function is evaluated."""
        return int(self.duration / self.resolution)

    @property
    def frames(self) -> Tensor:
        """
        The time frames at which the impulse response function is evaluated.

        Time frames start at `offset` and are spaced `resolution` apart, with shape `(1, num_frames)`, at
        :func:`keras.config.floatx` precision. Use :meth:`get_frames` to build them at another precision.

        """
        return self.get_frames()

    def get_frames(self, dtype: str | None = None) -> Tensor:
        """
        Build the time frames at which the impulse response function is evaluated.

        Parameters
        ----------
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Time frames of shape `(1, num_frames)` and dtype `dtype`, starting at `offset` and spaced
            `resolution` apart.

        Notes
        -----
        `duration` is an upper bound: the axis holds `num_frames = int(duration / resolution)` samples spaced
        exactly `resolution` apart, so it ends at the last whole sample at or below `duration` rather than at
        `duration` itself. At the defaults that is 32 samples from 0 to 31 seconds. Note that nilearn differs:
        ``spm_hrf(time_length=32)`` spans 0 to 32 seconds in 32 samples, so its spacing is `32 / 31` seconds
        rather than the requested 1.

        The axis is built at the requested precision rather than cast from a cached one, so a `float64`
        request carries `float64` time values and not widened `float32` ones. Results are cached per dtype.
        The cache is not invalidated if `duration`, `offset` or `resolution` are reassigned after
        construction.

        """
        dtype = get_dtype(dtype)

        if dtype not in self._frames:
            grid = ops.arange(self.num_frames, dtype=dtype) * self.resolution + self.offset
            self._frames[dtype] = ops.expand_dims(grid, 0)

        return self._frames[dtype]

    def _join_default_parameters(self, parameters: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.default_parameters, dict):
            parameters = parameters.copy()

            for key, val in self.default_parameters.items():
                if key not in parameters.columns:
                    parameters[key] = val

        self._check_parameters(parameters)

        return parameters

    @property
    @abstractmethod
    def _all_parameter_names(self) -> list[str]:
        """Names of all parameters used by the model, including ones covered by `default_parameters`."""

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters that must be supplied by the caller.

        Excludes names covered by :attr:`default_parameters`; those may still be overridden by supplying the
        corresponding column in ``parameters``.

        """
        default_names = set(self.default_parameters) if isinstance(self.default_parameters, dict) else set()
        return [name for name in self._all_parameter_names if name not in default_names]

    @doc
    @abstractmethod
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Compute the impulse response.

        Parameters
        ----------
        %(parameters)s Parameter values override default parameters.
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        Raises
        ------
        %(raises_missing_parameters)s

        """
