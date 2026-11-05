"""Tests for the backend compilation primitive."""

import numpy as np
from keras import ops
from prfmodel._backend import compile_fun
from prfmodel.typing import Tensor


class TestCompileFun:
    """Tests for the backend's compilation primitive.

    Every backend has one, so this is not skipped anywhere: `jax.jit` on JAX, `tf.function` on TensorFlow
    and `torch.compile` on PyTorch.

    """

    def test_compiling_preserves_the_result(self):
        """Test that compilation does not change what the function computes."""

        def double(value: Tensor) -> Tensor:
            return value * 2.0

        argument = ops.convert_to_tensor([1.0, 2.0])

        np.testing.assert_allclose(
            ops.convert_to_numpy(compile_fun(double)(argument)),
            ops.convert_to_numpy(double(argument)),
        )

    def test_a_compiled_function_is_reusable(self):
        """Test that the compiled function can be called more than once, which is the point of caching it."""

        def double(value: Tensor) -> Tensor:
            return value * 2.0

        compiled = compile_fun(double)

        first = ops.convert_to_numpy(compiled(ops.convert_to_tensor([1.0, 2.0])))
        second = ops.convert_to_numpy(compiled(ops.convert_to_tensor([3.0, 4.0])))

        np.testing.assert_allclose(first, [2.0, 4.0])
        np.testing.assert_allclose(second, [6.0, 8.0])
