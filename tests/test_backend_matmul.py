"""Tests for the backend matrix multiplication precision primitive."""

import keras
import numpy as np
import pytest
from keras import ops
from prfmodel._backend import full_precision_matmul


def _matmul_precision_setting() -> object:
    """Return the backend's current process-wide matmul precision setting."""
    match keras.backend.backend():
        case "tensorflow":
            import tensorflow as tf  # noqa: PLC0415 (only the active backend is installed)

            return tf.config.experimental.tensor_float_32_execution_enabled()
        case "jax":
            import jax  # noqa: PLC0415 (only the active backend is installed)

            return jax.config.jax_default_matmul_precision
        case "torch":
            import torch  # noqa: PLC0415 (only the active backend is installed)

            return torch.get_float32_matmul_precision()
        case other:
            pytest.fail(f"Backend '{other}' is not supported.")


class TestFullPrecisionMatmul:
    """Tests for the context manager that requests full float32 matrix multiplication precision."""

    def test_matches_float64_reference(self):
        """Test that a float32 matrix product inside the context agrees with a float64 reference."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((64, 300)).astype(np.float32)
        b = rng.standard_normal((300, 128)).astype(np.float32)

        with full_precision_matmul():
            result = ops.convert_to_numpy(ops.matmul(ops.convert_to_tensor(a), ops.convert_to_tensor(b)))

        np.testing.assert_allclose(result, a.astype(np.float64) @ b.astype(np.float64), rtol=1e-4, atol=1e-4)

    def test_restores_previous_setting(self):
        """Test that the setting in effect before the context is in effect again after it."""
        before = _matmul_precision_setting()

        with full_precision_matmul():
            pass

        assert _matmul_precision_setting() == before

    def test_restores_previous_setting_on_error(self):
        """Test that the previous setting is restored even when the body of the context raises."""
        before = _matmul_precision_setting()

        with pytest.raises(RuntimeError), full_precision_matmul():
            raise RuntimeError

        assert _matmul_precision_setting() == before
