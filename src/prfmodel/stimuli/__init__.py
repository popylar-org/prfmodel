"""Containers that contain stimuli information as model input.

This module contains stimuli classes that serve as input for models. Currently, only stimuli for population receptive
field (pRF) models, connective field (CF), and contrast sensitivity function (CSF) models are implemented.

Each stimulus type has a user-facing class (i.e., :class:``PRFStimulus``, :class:``CFStimulus``, and ``CSFStimulus``)
that holds :class:``numpy.array`` objects. Internally, these are converted to counterparts that hold backend tensor
objects to enable backend-specific optimization (i.e., :class:``PRFStimulusTensors``, :class:``CFStimulusTensors``,
and ``CSFStimulusTensors``).

:class:``Stimulus`` is an abstract base class that the user-facing stimulus classes inherit
from. Subclasses must implement the abstract :meth:``Stimulus.to_tensors`` method to determine how non-tensor
objects are converted to tensors.

"""

from ._cf import CFStimulus
from ._cf import CFStimulusTensors
from ._csf import CSFStimulus
from ._csf import CSFStimulusTensors
from ._prf import PRFStimulus
from ._prf import PRFStimulusTensors
from .base import Stimulus
from .base import StimulusTensors

__all__ = [
    "CFStimulus",
    "CFStimulusTensors",
    "CSFStimulus",
    "CSFStimulusTensors",
    "PRFStimulus",
    "PRFStimulusTensors",
    "Stimulus",
    "StimulusTensors",
]
