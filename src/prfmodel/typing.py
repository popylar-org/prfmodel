"""Multi-backend variable types."""

from typing import TypeAlias
import keras

match keras.backend.backend():
    case "jax":
        from jax import Array as BackendTensor  # type: ignore[assignment]
    case "tensorflow":
        from tensorflow import Tensor as BackendTensor  # type: ignore[assignment]
    case "torch":
        from torch import Tensor as BackendTensor  # type: ignore[assignment]
    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)

Tensor: TypeAlias = BackendTensor
"""Backend-specific tensor type."""
