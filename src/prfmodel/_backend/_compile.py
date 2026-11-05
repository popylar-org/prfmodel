"""Backend-specific compilation primitives.

Every backend can compile a Python function into a graph, but each does it with its own primitive. This
module is the single place where that difference is resolved, so that the rest of the package stays
written against :mod:`keras.ops` alone.

"""

from collections.abc import Callable
import keras

match keras.backend.backend():
    case "tensorflow":
        import tensorflow as tf

        def compile_fun(fun: Callable) -> Callable:
            """Compile a function into a TensorFlow graph."""
            return tf.function(fun)

    case "jax":
        import jax

        def compile_fun(fun: Callable) -> Callable:
            """Compile a function with the JAX just-in-time compiler."""
            return jax.jit(fun)

    case "torch":
        import torch

        def compile_fun(fun: Callable) -> Callable:
            """Compile a function with the PyTorch just-in-time compiler."""
            return torch.compile(fun)

    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)
