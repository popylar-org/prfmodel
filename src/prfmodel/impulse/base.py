"""Impulse model base classes.

Classes in this module inherit from :class:`~prfmodel.protocols.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.protocols.ModelProtocol.parameter_names` property.

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

The user-facing :meth:`__call__` returns a :class:`numpy.ndarray`, while ``call`` returns a backend tensor.
Use ``call`` when a backend tensor is required, for example inside a fitter or another model's ``call``.

Impulse models can have default parameters that are defined during initialization. The default parameters are
added to the user-supplied parameter dataframe (replicating the default value for each unit) in the `__call__`
method if not already present.

"""

from abc import abstractmethod
from typing import TypeVar
import numpy as np
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.protocols import ModelProtocol
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import _get_norm_fun
from prfmodel.utils import as_tensor_frame
from prfmodel.utils import get_dtype

P = TypeVar("P", pd.DataFrame, TensorFrame)
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
        Dictionary keys must be valid parameter names. Default values are overridden by user-supplied parameters in
        the :meth:`__call__` method.

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom response
    models. Subclasses must override the abstract :attr:`parameter_names` property and the :meth:`call`
    method.

    `duration` and `resolution` must be positive, checked once at construction. `offset` is
    unconstrained and may be negative: the densities are zero below their support, so frames at or below
    zero contribute zero. A negative `offset` therefore gives the kernel leading zeros, which delays the
    convolved response by `-offset / resolution` frames without changing a `"sum"` normalization. Note
    that `num_frames` is derived from `duration` alone, so a negative `offset` shifts the sampling
    window rather than widening it, and the far tail of the response is truncated by the same amount.

    Each frame is sampled at the centre of the interval it stands for, not at its leading edge: frame
    `i` covers `[offset + i * resolution, offset + (i + 1) * resolution)` and is evaluated at its
    midpoint, `offset + (i + 0.5) * resolution`. A sample represents the whole interval, and it keeps
    `t = 0` off the axis. At the defaults that is 32 samples centred at 0.5, 1.5, ..., 31.5 seconds.

    `duration` is an upper bound: the time frames array holds `num_frames = int(duration / resolution)` samples
    spaced exactly `resolution` apart, so it ends at the last whole sample at or below `duration` rather than at
    `duration` itself.

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

        # The time axis is fixed at construction, so it is checked once here rather than on every
        # evaluation. 'offset' is deliberately unconstrained: the densities are zero below their
        # support, so a frame at or below zero contributes zero rather than a 'NaN'.
        if duration <= 0.0:
            msg = f"Argument 'duration' must be > 0 but is {duration}"
            raise ValueError(msg)

        if resolution <= 0.0:
            msg = f"Argument 'resolution' must be > 0 but is {resolution}"
            raise ValueError(msg)

        self._duration = duration
        self._offset = offset
        self._resolution = resolution

        # Check if norm arg is valid
        if norm is not None:
            _get_norm_fun(norm)

        self.norm = norm

        if isinstance(default_parameters, dict):
            # Reads the subclass declaration directly, which is safe only because it is a literal:
            # 'self.default_parameters' is not assigned until below, and 'parameter_names' would read it.
            if any(key not in self.parameter_names for key in default_parameters):
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

    @doc
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

    @doc
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> np.ndarray:
        """
        Compute the impulse response.

        This is the public entry point. It merges :attr:`default_parameters` into `parameters`, validates the
        result, converts it to tensors and delegates to :meth:`call`, then returns the result as a
        :class:`numpy.ndarray`. Subclasses implement :meth:`call`. Use :meth:`call` when a backend tensor is
        required.

        Parameters
        ----------
        %(parameters)s Parameter values override default parameters.
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d_array)s

        Raises
        ------
        %(raises_missing_parameters)s

        """
        dtype = get_dtype(dtype)
        parameters = self._join_default_parameters(parameters)
        self.check_parameter_names(parameters)
        self.check_parameter_values(parameters)

        return ops.convert_to_numpy(self.call(as_tensor_frame(parameters[self.parameter_names], dtype)))

    @doc
    def check_parameter_names(self, parameters: pd.DataFrame) -> None:
        """Check that required parameter names are supplied.

        If not already present, adds default parameter columns to ``parameters``.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter name in the :attr:`parameters_names` attribute is not a column in ``parameters``.

        """
        super().check_parameter_names(self._join_default_parameters(parameters))

    @doc
    def check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the model is defined on.

        If not already present, adds default parameter columns to ``parameters``.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When a parameter that must be ``> 0`` is zero or negative.

        """
        super().check_parameter_values(self._join_default_parameters(parameters))

    @doc
    @abstractmethod
    def call(self, parameters: TensorFrame) -> Tensor:
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
