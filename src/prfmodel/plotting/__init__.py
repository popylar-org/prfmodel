"""Plotting functions.

Contains miscallenous plotting and visualization functions. Currently, only functions for visualizing population
receptive field (pRF) stimuli are implemented.

"""

from ._stimuli import animate_2d_prf_stimulus
from ._stimuli import plot_2d_prf_stimulus

__all__ = [
    "animate_2d_prf_stimulus",
    "plot_2d_prf_stimulus",
]
