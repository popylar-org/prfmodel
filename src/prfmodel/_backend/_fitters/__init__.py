"""Backend-specific fitters."""

import keras

match keras.backend.backend():
    case "jax":
        from .jax import JAXSGDFitter as BackendSGDFitter
    case "tensorflow":
        from .tensorflow import TensorFlowSGDFitter as BackendSGDFitter
    case "torch":
        from .torch import TorchSGDFitter as BackendSGDFitter
    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)

__all__ = [
    "BackendSGDFitter",
]
