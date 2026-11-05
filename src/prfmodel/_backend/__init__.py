"""Backend-specific external and internal imports.

`BackendSGDFitter` is deliberately *not* re-exported here. It lives in
:mod:`prfmodel._backend._fitters`, whose base class imports :func:`compile_fun` from this package.
Importing it here would close that cycle.

"""

from ._compile import compile_fun
from ._external import gammaln

__all__ = [
    "compile_fun",
    "gammaln",
]
