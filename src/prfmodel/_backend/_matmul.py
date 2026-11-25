"""Backend-specific control of float32 matrix multiplication precision.

On recent NVIDIA GPUs (Ampere and newer), float32 matrix multiplications can run in TensorFloat-32 (TF32), which rounds
the inputs to about three significant digits. JAX and TensorFlow do so by default, PyTorch can be configured to. This
module is the single place where full float32 precision is requested, so that the rest of the package stays written
against :mod:`keras.ops` alone.

Element-wise operations and float64 are never affected, and neither is any computation on the CPU.

"""

from collections.abc import Iterator
from contextlib import contextmanager
import keras

match keras.backend.backend():
    case "tensorflow":
        import tensorflow as tf

        @contextmanager
        def full_precision_matmul() -> Iterator[None]:
            """Disable TF32 for float32 matrix multiplications inside the context.

            TensorFlow has no per-operation precision, only a process-wide switch, so it is turned off on entry and
            the previous setting is restored on exit. The switch is read when a kernel runs, so it also applies to
            functions compiled with :func:`tf.function`. Because the switch is global, it also affects other
            threads while the context is active.

            """
            previous = tf.config.experimental.tensor_float_32_execution_enabled()
            tf.config.experimental.enable_tensor_float_32_execution(False)
            try:
                yield
            finally:
                tf.config.experimental.enable_tensor_float_32_execution(previous)

    case "jax":
        import jax

        @contextmanager
        def full_precision_matmul() -> Iterator[None]:
            """Use the highest matrix multiplication precision for operations traced inside the context.

            The setting is part of the JAX trace context, so a function compiled with :func:`jax.jit` must be traced
            inside the context to pick it up, and is traced again if it is later called outside of it.

            """
            with jax.default_matmul_precision("highest"):
                yield

    case "torch":
        import torch

        @contextmanager
        def full_precision_matmul() -> Iterator[None]:
            """Use the highest float32 matrix multiplication precision inside the context.

            PyTorch already defaults to `"highest"`, so this only matters when a user lowered it. The previous
            setting is restored on exit. Like on TensorFlow, the setting is process-wide.

            """
            previous = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
            try:
                yield
            finally:
                torch.set_float32_matmul_precision(previous)

    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)
