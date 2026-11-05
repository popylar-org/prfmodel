"""Density functions.

This modules contains helper functions to compute densities: gamma densities that are used in the
:py:mod:`~prfmodel.impulse` module and normal densities that are used in the :py:mod:`~prfmodel.models` module.

"""

from ._gamma import derivative_gamma_density
from ._gamma import gamma_density
from ._gamma import shifted_gamma_density
from ._gaussian import normal_density

__all__ = [
    "derivative_gamma_density",
    "gamma_density",
    "normal_density",
    "shifted_gamma_density",
]
