"""Stimulus base classes."""

from abc import abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class StimulusTensors:
    """Base class for the tensor-side counterpart of a :class:`~prfmodel.stimuli.Stimulus`.

    A :class:`~prfmodel.stimuli.Stimulus` holds :class:`numpy.ndarray` fields and validates them; it is what
    users build and what plotting and I/O work with. A `StimulusTensors` holds the same fields converted to
    backend tensors and validates nothing. It is what a model's tensor-only
    :meth:`~prfmodel.models.base.BaseTuning.call` receives, so that the method touches no
    :mod:`numpy` and can be traced by a backend compiler.

    Build one with :meth:`~prfmodel.stimuli.Stimulus.to_tensors`. Subclasses declare the array fields of the
    stimulus they mirror; fields that no model reads, such as
    :attr:`~prfmodel.stimuli.PRFStimulus.dimension_labels`, are deliberately left out.

    Notes
    -----
    A `StimulusTensors` is built fresh by every :meth:`~prfmodel.stimuli.Stimulus.to_tensors` call and must
    not be cached on the stimulus it came from. A tensor created while a function is being traced belongs to
    that trace and cannot be read from a later one, so a cached bundle would leak the first trace into every
    subsequent one. Callers that want to pay the conversion once should hoist `to_tensors` themselves, outside
    the region being compiled, which is what the fitters do.

    """

    # Contains tensors as attributes which are not hashable
    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True, eq=False)
class Stimulus:
    """Stimulus base class."""

    # Contains numpy arrays as attributes which are not hashable
    __hash__ = None  # type: ignore[assignment]

    @abstractmethod
    def to_tensors(self, dtype: str | None = None) -> StimulusTensors:
        """Convert the stimulus arrays into backend tensors.

        Parameters
        ----------
        dtype : str, optional
            The dtype to convert the stimulus arrays to. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        StimulusTensors
            The stimulus arrays as tensors, ready to be passed to a model's
            :meth:`~prfmodel.models.base.BaseTuning.call`.

        """

    def __repr__(self) -> str:
        """Create a round-trippable string representation of the stimulus object."""
        arg_list = []

        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                arg_list.append(f"{key}={np.array_repr(val)}")
            else:
                arg_list.append(f"{key}={val!r}")

        return f"{self.__class__.__name__}({', '.join(arg_list)})"

    def __str__(self) -> str:
        """Create a human-readable string representation of the stimulus object."""
        str_list = []

        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                arr_shape = ", ".join([str(s) for s in val.shape])
                str_list.append(f"{key}=array[{arr_shape}]")
            else:
                str_list.append(f"{key}={val}")

        return f"{self.__class__.__name__}({', '.join(str_list)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for (key_self, val_self), (key_other, val_other) in zip(
            self.__dict__.items(),
            other.__dict__.items(),
            strict=True,
        ):
            if key_self != key_other:
                return False
            if isinstance(val_self, np.ndarray) and isinstance(val_other, np.ndarray):
                if val_self.shape != val_other.shape:
                    return False
                if not np.all(val_self == val_other):
                    return False
            elif val_self != val_other:
                return False

        return True
