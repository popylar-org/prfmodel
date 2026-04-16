"""Multi-backend variable types.

This module contains package specific types that are used by static type checkers (e.g., mypy).

"""

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
"""Backend-specific tensor type.

This is a type alias (:class:`typing.TypeAlias`). Depending on the Keras backend the alias refers to a different type.
For the TensorFlow backend, it refers to :class:`tensorflow.Tensor`. For the PyTorch backend, it refers to
:class:`torch.Tensor`. For the JAX backend, it refers to :class:`jax.Array`.

"""
