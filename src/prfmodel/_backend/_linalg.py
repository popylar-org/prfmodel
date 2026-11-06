"""Backend-specific linear algebra.

Keras provides the linear algebra functions, but a backend can fail for some inputs on some devices. This module
works around such failures, so that the rest of the package can call these functions without device checks.

"""

from typing import cast
import keras
from keras import ops
from prfmodel.typing import Tensor

match keras.backend.backend():
    case "torch":
        import torch

        def svd(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """Compute the reduced singular value decomposition of a (batch of) matrices.

            On Apple MPS, the decomposition is computed on the CPU and its factors are moved back to MPS. The
            MPS implementation in torch >= 2.14 fails to converge for some valid matrices (``linalg.svd: The
            algorithm failed to converge``), e.g., when a column is nearly but not exactly zero, which happens
            for a pRF that barely overlaps the stimulus. The matrices passed here are small, so the copy is cheap.

            """
            x_torch = cast("torch.Tensor", x)

            if x_torch.device.type != "mps":
                return ops.linalg.svd(x, full_matrices=False)

            # torch directly rather than keras.ops, which would move the CPU copy back to the default device
            u_factor, singular_values, vt_factor = torch.linalg.svd(x_torch.cpu(), full_matrices=False)
            device = x_torch.device

            return cast(
                "tuple[Tensor, Tensor, Tensor]",
                (u_factor.to(device), singular_values.to(device), vt_factor.to(device)),
            )

    case "tensorflow" | "jax":

        def svd(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            """Compute the reduced singular value decomposition of a (batch of) matrices."""
            return ops.linalg.svd(x, full_matrices=False)

    case other:
        msg = f"Backend '{other}' is not supported."
        raise ValueError(msg)
