"""Backend-specific imports.

Imports functions and classes that are not implemented in Keras but in all backends.

"""

import keras

match keras.backend.backend():
    case "tensorflow":
        from tensorflow.math import lgamma as gammaln
    case "torch":
        from torch.special import gammaln
    case "jax":
        from jax.scipy.special import gammaln  # noqa: F401 (unused import)
