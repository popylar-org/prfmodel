"""Stimulus base classes."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class Stimulus:
    """Stimulus base class."""

    # Contains numpy arrays as attributes which are not hashable
    __hash__ = None  # type: ignore[assignment]

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
