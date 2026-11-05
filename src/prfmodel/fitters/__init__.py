"""Fit models to data and estimate model parameters.

This module contains classes for fitting models implemented in the :mod:`~prfmodel.models` module to data.

Currently, only three fitting methods are available: Grid search, least-squares, and stochastic gradient descent (SGD).

The fitting methods can be combined: For examples, the parameter estimates from the grid search can be augmented
with least-squares estimates or used as the starting point for SGD for finetuning the estimates. See
:ref:`tutorials` for details.

Each fitting method returns a history object that stores final loss scores for all data units.

The functions :func:`~prfmodel.fitters.grid_values_2d_prf` and :func:`~prfmodel.fitters.grid_values_1d_prf` create
values for the pRF center and size parameters of a :class:`~prfmodel.fitters.GridFitter` from the stimulus.

The :mod:`~prfmodel.fitters.adapter` submodule contains functionality to transform parameters during model fitting
(e.g., to optimize a parameter on the log scale). Currently, this is only implemented for SGD.

The :mod:`~prfmodel.fitters.losses` submodule contains additional loss functions, such as
:class:`~prfmodel.fitters.losses.CorrelationLoss`, which is the default loss of
:class:`~prfmodel.fitters.GridFitter`.

"""

from ._grid import GridFitter
from ._grid import GridHistory
from ._grid_values import GridAlignmentWarning
from ._grid_values import grid_values_1d_prf
from ._grid_values import grid_values_2d_prf
from ._least_squares import LeastSquaresFitter
from ._least_squares import LeastSquaresHistory
from ._sgd import SGDFitter
from ._sgd import SGDHistory

__all__ = [
    "GridAlignmentWarning",
    "GridFitter",
    "GridHistory",
    "LeastSquaresFitter",
    "LeastSquaresHistory",
    "SGDFitter",
    "SGDHistory",
    "grid_values_1d_prf",
    "grid_values_2d_prf",
]
