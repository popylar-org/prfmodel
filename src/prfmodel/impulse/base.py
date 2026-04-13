"""Impulse model base classes."""

from abc import abstractmethod
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol
from prfmodel.utils import _get_norm_fun


class BaseImpulse(ModelProtocol):
    """
    Abstract base class for impulse response models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom impulse response models.
    Subclasses must override the abstract :meth:`__call__` method.

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
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Keys must be valid parameter names.

    """

    def __init__(
        self,
        duration: float = 32.0,
        offset: float = 0.0001,
        resolution: float = 1.0,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | None = None,
    ):
        super().__init__()

        self.duration = duration
        self.offset = offset
        self.resolution = resolution

        # Check if norm arg is valid
        if norm is not None:
            _get_norm_fun(norm)

        self.norm = norm

        if default_parameters is not None:
            if any(key not in self.parameter_names for key in default_parameters):
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
        if self.default_parameters is not None:
            parameters = parameters.copy()

            for key, val in self.default_parameters.items():
                parameters[key] = val

        return parameters

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """A list with names of parameters that are used by the model."""

    @doc
    @abstractmethod
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Compute the impulse response.

        Parameters
        ----------
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
