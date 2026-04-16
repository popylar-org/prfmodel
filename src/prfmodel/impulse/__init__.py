"""Impulse response submodels.

This module contains impulse response models that describe the physiological shape of neural response signals. For
example, in functional magnetic resonance imaging (fMRI), the physiological shape of the neural signal is captured by
the hemodynamic response function (HRF). Importantly, the impulse models implemented in this module are agnostic to the
neuroimaging method. Depending on the (default) parameters given by the user, they can be adapted to different
physiological shapes.

Impulse models are intended to be used as submodels within canonical models, e.g.,
:class:`~prfmodel.models.prf.canonical.CanonicalPRFModel`.

The function :func:`~prfmodel.impulse.convolve_prf_impulse_response` can be used to convolve an impulse response with
a response from a population receptive field (pRF) response model.

"""

from ._convolve import convolve_prf_impulse_response
from ._shifted_gamma import ShiftedGammaImpulse
from ._two_gamma import TwoGammaImpulse
from ._two_gamma_deriv import DerivativeTwoGammaImpulse

__all__ = [
    "DerivativeTwoGammaImpulse",
    "ShiftedGammaImpulse",
    "TwoGammaImpulse",
    "convolve_prf_impulse_response",
]
