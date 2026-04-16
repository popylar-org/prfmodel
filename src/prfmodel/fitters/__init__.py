"""Fit models to data and estimate model parameters.

This module contains classes for fitting models implemented in the :mod:`~prfmodel.models` module to data.

Currently, only three fitting methods are available: Grid search, least-squares, and stochastic gradient descent (SGD).

The fitting methods can be combined: For examples, the parameter estimates from the grid search can be augmetented
with least-squares estimates or used as the starting point for SGD for finetuning the estimates. See
:ref:`tutorials` for details.

Each fitting method returns a history object that stores final loss scores for all data units.

The :mod:`~prfmodel.fitters.adapter` submodule contains functionality to transform parameters during model fitting
(e.g., to optimize a parameter on the log scale). Currently, this is only implemented for SGD.

"""

from ._grid import GridFitter
from ._grid import GridHistory
from ._least_squares import LeastSquaresFitter
from ._least_squares import LeastSquaresHistory
from ._sgd import SGDFitter
from ._sgd import SGDHistory

__all__ = [
    "GridFitter",
    "GridHistory",
    "LeastSquaresFitter",
    "LeastSquaresHistory",
    "SGDFitter",
    "SGDHistory",
]
