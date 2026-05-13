"""Containers that contain stimuli information as model input.

This module contains stimuli classes that serve as input for models. Currently, only stimuli for population receptive
field (pRF) models and connective field (CF) models are implemented.

"""

from ._cf import CFStimulus
from ._prf import PRFStimulus
from .base import Stimulus

__all__ = [
    "CFStimulus",
    "PRFStimulus",
    "Stimulus",
]
