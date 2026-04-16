"""Scaling submodels.

This module contains models that scale (and shift) predicted neural responses.

Scaling models are intended to be used as submodels within canonical models, e.g.,
:class:`~prfmodel.models.prf.canonical.CanonicalPRFModel`.

"""

from ._amplitude import BaselineAmplitude
from ._div_norm_amplitude import DivNormAmplitude
from ._dog_amplitude import DoGAmplitude

__all__ = [
    "BaselineAmplitude",
    "DivNormAmplitude",
    "DoGAmplitude",
]
