"""Plotting functions.

Contains plotting and visualization functions for stimuli, observed and predicted responses, model evaluation, and
estimated model parameters. Functions that create a single plot return the figure and axes objects and most accept
an existing `ax` so that they can be combined in a larger figure.

"""

from ._evaluation import plot_r_squared_comparison
from ._evaluation import plot_r_squared_hist
from ._parameters import plot_2d_prf_centers
from ._parameters import plot_grid_parameter_distribution
from ._parameters import plot_parameter_by_roi
from ._responses import plot_observed_predicted
from ._responses import plot_response_heatmap
from ._stimuli import animate_2d_prf_stimulus
from ._stimuli import plot_1d_prf_stimulus
from ._stimuli import plot_2d_prf_stimulus
from ._stimuli import plot_csf_stimulus_curve
from ._stimuli import plot_csf_stimulus_design
from ._surface import plot_surface_stat_map

__all__ = [
    "animate_2d_prf_stimulus",
    "plot_1d_prf_stimulus",
    "plot_2d_prf_centers",
    "plot_2d_prf_stimulus",
    "plot_csf_stimulus_curve",
    "plot_csf_stimulus_design",
    "plot_grid_parameter_distribution",
    "plot_observed_predicted",
    "plot_parameter_by_roi",
    "plot_r_squared_comparison",
    "plot_r_squared_hist",
    "plot_response_heatmap",
    "plot_surface_stat_map",
]
