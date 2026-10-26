"""Documentation about prfmodel."""

import logging
from prfmodel._backend_check import check_backend

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Every module imports keras, so this must run first to replace a failed backend import with installation instructions
check_backend()

__version__ = "0.1.0"


__all__ = []
