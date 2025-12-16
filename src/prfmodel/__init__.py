"""Documentation about prfmodel."""

import logging
from prfmodel.stimulus.prf import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.0.1"


__all__ = [
    "PRFStimulus",
    "Tensor",
    "convert_parameters_to_tensor",
]
