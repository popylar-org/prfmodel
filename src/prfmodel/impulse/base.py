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

        self._frames: Tensor | None = None

    @property
    def num_frames(self) -> int:
        """The total number of time frames at which the impulse response function is evaluated."""
        return int(self.duration / self.resolution)

    @property
    def frames(self) -> Tensor:
        """
        The time frames at which the impulse response function is evaluated.

        Time frames are linearly interpolated between `offset` and `duration` and have shape `(1, num_frames)`.

        """
        if self._frames is None:
            self._frames = ops.expand_dims(ops.linspace(self.offset, self.duration, self.num_frames), 0)

        return self._frames

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
