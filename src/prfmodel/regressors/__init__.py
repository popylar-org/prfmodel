"""Regressor submodels.

This module contains regressor models that add a linear combination of design time courses (with per-unit beta
weights) to a canonical model prediction. They are useful for nuisance or task regressors in a general-linear-model
style analysis.

Two types of regressor models are currently implemented:

* :class:`AdditiveRegressors` adds the regressors directly to the prediction.
* :class:`ConvolvedRegressors` convolves each regressor with an impulse response before adding.

A :class:`RegressorsList` composite is used internally when multiple regressor models are passed to a canonical
model.

At call time the regressor design data is provided as a :class:`pandas.DataFrame` (or a list of DataFrames for a
:class:`RegressorsList`). Each regressor model looks up the columns it needs by name, so column order is
unimportant and extra columns are silently ignored.

Regressor models are intended to be used as submodels within canonical models, e.g.,
:class:`~prfmodel.models.prf.canonical.CanonicalPRFModel`.

"""

from ._additive import AdditiveRegressors
from ._convolved import ConvolvedRegressors
from ._list import RegressorsList

__all__ = [
    "AdditiveRegressors",
    "ConvolvedRegressors",
    "RegressorsList",
]
