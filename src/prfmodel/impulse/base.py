"""Impulse model base classes.

Classes in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example, :class:`~prfmodel.models.base.BaseImpulse` defines that all child classes
must implement a :meth:`~prfmodel.models.base.BaseImpulse.call` method that takes a set of parameters
as input. However, it leaves it up to each child class to define how input parameters are used to make
model predictions.

All base classes have a concrete user-facing :meth:``__call__`` method
(e.g., :meth:`~prfmodel.impulse.base.BaseImpulse.__call__`) that takes non-tensor arguments and
performs validation checks. This method calls the abstract ``call`` method that must be implemented by each child
class and only accepts tensor arguments to enable backend compilation.

"""

from abc import abstractmethod
from typing import ClassVar
from typing import TypeVar
import numpy as np
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol
from prfmodel.utils import ParamsDict
from prfmodel.utils import _get_norm_fun
from prfmodel.utils import as_params
from prfmodel.utils import get_dtype

P = TypeVar("P", pd.DataFrame, ParamsDict)
"""Either representation of a parameter table: the user-facing data frame or its tensor-holding counterpart."""


class BaseImpulse(ModelProtocol):
    """
    Abstract base class for impulse models.

    An impulse model takes a set of parameters as input a predicts an impulse for time frames that are
    defined by an offset, duration, and resolution.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0
        The offset of the impulse response (in seconds).
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

    `duration`, `offset` and `resolution` are read-only: they define the time axis that
    :meth:`get_frames` caches, so a model that needs a different axis is constructed anew rather than
    modified in place.

    """

    def __init__(
        self,
        duration: float = 32.0,
        offset: float = 0.0,
        resolution: float = 1.0,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | str | None = None,
    ):
        super().__init__()

        self._duration = duration
        self._offset = offset
        self._resolution = resolution

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

        self._frames: dict[str, np.ndarray] = {}

    @property
    def duration(self) -> float:
        """The duration of the impulse response (in seconds). Read-only."""
        return self._duration

    @property
    def offset(self) -> float:
        """The offset of the impulse response (in seconds). Read-only."""
        return self._offset

    @property
    def resolution(self) -> float:
        """The time resolution of the impulse response (in seconds). Read-only."""
        return self._resolution

    @property
    def num_frames(self) -> int:
        """The total number of time frames at which the impulse response function is evaluated."""
        return int(self.duration / self.resolution)

    def get_frames(self, dtype: str | None = None) -> Tensor:
        """
        Build the time frames at which the impulse response function is evaluated.

        Parameters
        ----------
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Time frames of shape `(1, num_frames)` and dtype `dtype`. The first frame is at
            `offset + resolution / 2` and frames are spaced `resolution` apart.

        Notes
        -----
        Each frame is sampled at the centre of the interval it stands for, not at its leading edge: frame
        `i` covers `[offset + i * resolution, offset + (i + 1) * resolution)` and is evaluated at its
        midpoint, `offset + (i + 0.5) * resolution`. A sample represents the whole interval, and it keeps
        `t = 0` off the axis. At the defaults that is 32 samples centred at 0.5, 1.5, ..., 31.5 seconds.

        `duration` is an upper bound: the time frames array holds `num_frames = int(duration / resolution)` samples
        spaced exactly `resolution` apart, so it ends at the last whole sample at or below `duration` rather than at
        `duration` itself.

        The time frames are cached as a `numpy.ndarray` rather than as a backend tensor, and converted on every
        call.

        """
        dtype = get_dtype(dtype)

        if dtype not in self._frames:
            steps = np.arange(self.num_frames, dtype=dtype)
            self._frames[dtype] = np.expand_dims(steps * self.resolution + self.resolution / 2 + self.offset, 0)

        return ops.convert_to_tensor(self._frames[dtype], dtype=dtype)

    def _join_default_parameters(self, parameters: P) -> P:
        if not isinstance(self.default_parameters, dict):
            return parameters

        missing = {key: val for key, val in self.default_parameters.items() if key not in parameters.columns}

        if not missing:
            return parameters

        parameters = parameters.copy()

        for key, val in missing.items():
            parameters[key] = val

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
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Compute the impulse response.

        This is the public entry point. It merges :attr:`default_parameters` into `parameters`, validates the
        result, converts it to tensors and delegates to :meth:`call`. Subclasses implement :meth:`call`.

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
        dtype = get_dtype(dtype)
        parameters = self._join_default_parameters(parameters)
        self._check_parameters(parameters)
        self._check_parameter_values(parameters)

        return self.call(as_params(parameters, dtype))

    _positive_parameter_names: ClassVar[tuple[str, ...]] = ()
    """Parameters that must be strictly positive for the impulse response to be defined."""

    def _check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the impulse response is defined on.

        Called from :meth:`__call__`, which is never traced, so the values here are always concrete and the
        check always runs.

        Subclasses declare the parameters this applies to with :attr:`_positive_parameter_names`.

        """
        for name in self._positive_parameter_names:
            if name in parameters.columns and not (parameters[name].to_numpy() > 0.0).all():
                msg = f"Parameter '{name}' must be > 0"
                raise ValueError(msg)

    @doc
    @abstractmethod
    def call(self, parameters: ParamsDict) -> Tensor:
        """
        Compute the impulse response from tensors.

        Parameters
        ----------
        %(parameters_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        Notes
        -----
        Implementations must begin by calling :meth:`_join_default_parameters`, because a canonical model
        reaches its impulse submodel through this method rather than through :meth:`__call__`.

        Implementations must also be traceable by a backend compiler. See
        :meth:`~prfmodel.models.base.BasePopulationResponse.call`.

        """
