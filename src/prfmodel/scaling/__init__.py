"""Scaling submodels.

This module contains models that scale (and shift) predicted neural responses. More complex scaling models also define
how responses from parallel submodels are integrated (e.g., :class:`~prfmodel.scaling.DoGAmplitude`).

Scaling models are intended to be used as submodels within canonical models, e.g.,
:class:`~prfmodel.models.prf.canonical.CanonicalPRFModel`.

"""

from ._amplitude import BaselineAmplitude
from ._delayed_gain_norm import DelayedGainNormScaling
from ._div_norm_amplitude import DivNormAmplitude
from ._dog_amplitude import DoGAmplitude

__all__ = [
    "BaselineAmplitude",
    "DelayedGainNormScaling",
    "DivNormAmplitude",
    "DoGAmplitude",
]
