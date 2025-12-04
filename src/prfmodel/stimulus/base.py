"""Stimulus base classes."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, eq=False)
class Stimulus:
    """Stimulus base class."""

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
        if not isinstance(other, type(self)):
            class_name = self.__class__.__name__
            msg = f"{class_name} objects can only be compared against other {class_name} objects"
            raise TypeError(msg)

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

    def __hash__(self):
        return hash(val for val in self.__dict__.values())
