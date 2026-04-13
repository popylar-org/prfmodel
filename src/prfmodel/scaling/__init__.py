"""Scaling models."""

from .amplitude import BaselineAmplitude
from .div_norm_amplitude import DivNormAmplitude
from .dog_amplitude import DoGAmplitude

__all__ = [
    "BaselineAmplitude",
    "DivNormAmplitude",
    "DoGAmplitude",
]
