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
        from keras import ops

        # Apple's Metal shading language allows at most 31 buffer arguments per kernel. Inductor does not
        # enforce this when it fuses operations for the MPS device, so a fused kernel with many inputs fails to
        # compile ('number of constant buffers exceeds maximum supported (31)'). Capping the number of unique
        # input and output buffers per fused kernel keeps them below the limit, with headroom for the extra
        # buffers a kernel uses for, e.g., its sizes.
        _MPS_MAX_FUSION_IO_BUFFERS = 24

        def compile_fun(fun: Callable) -> Callable:
            """Compile a function with the PyTorch just-in-time compiler.

            Inductor's layout optimization is disabled. Its heuristic inspects the `groups` argument of every
            convolution, and the depthwise convolution with the impulse response has one group per unit. When a
            fitter calls the compiled function with a new batch size, PyTorch recompiles it with dynamic shapes,
            `groups` becomes symbolic and the heuristic fails on accelerators (``TypeError: '>' not supported
            between instances of 'Node' and 'int'``, seen with torch 2.15 on Apple MPS). The optimization only
            changes the memory layout of convolution inputs, which does not benefit this 1D convolution.

            On Apple MPS, the number of buffers per fused kernel is also capped (see
            `_MPS_MAX_FUSION_IO_BUFFERS`).

            """
            options: dict = {"layout_optimization": False}

            # Keras places new tensors on its current device, so a probe tensor tells where the function will run
            if ops.convert_to_tensor(0.0).device.type == "mps":
                options["max_fusion_unique_io_buffers"] = _MPS_MAX_FUSION_IO_BUFFERS

            return torch.compile(fun, options=options)

    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)
