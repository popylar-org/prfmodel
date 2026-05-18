"""Density functions.

This modules contains helper functions to compute densities. Currently, it only contains gamma densities that are
used in the :py:mod:`~prfmodel.impulse` module.

"""

from ._gamma import derivative_gamma_density
from ._gamma import gamma_density
from ._gamma import shifted_gamma_density

__all__ = [
    "derivative_gamma_density",
    "gamma_density",
    "shifted_gamma_density",
]
