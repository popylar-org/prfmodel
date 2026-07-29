"""Plotting functions.

Contains miscallenous plotting and visualization functions. Currently, only functions for visualizing population
receptive field (pRF) and contrast sensitivity function (CSF) stimuli are implemented.

"""

from ._stimuli import animate_2d_prf_stimulus
from ._stimuli import plot_2d_prf_stimulus
from ._stimuli import plot_csf_stimulus_curve
from ._stimuli import plot_csf_stimulus_design

__all__ = [
    "animate_2d_prf_stimulus",
    "plot_2d_prf_stimulus",
    "plot_csf_stimulus_curve",
    "plot_csf_stimulus_design",
]
