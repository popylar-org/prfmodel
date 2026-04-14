"""Scaling models."""

from ._amplitude import BaselineAmplitude
from ._div_norm_amplitude import DivNormAmplitude
from ._dog_amplitude import DoGAmplitude

__all__ = [
    "BaselineAmplitude",
    "DivNormAmplitude",
    "DoGAmplitude",
]
