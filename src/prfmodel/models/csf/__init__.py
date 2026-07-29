"""Contrast sensitivity function (CSF) models.

This module contain a CSF response model that predicts a response to a stimulus containing contrast and spatial
frequencies. Currently, only the asymmetric log-parabolic (aLP) CSF is implemented in
:class:`CSFResponse`.

It also contains a canonical model that combines a CSF response with an impulse and a scaling model
(:class:`CanonicalCSFModel`) [1]_. In contrast to pRF and connective field models,
canonical CSF models do not require stimulus encoding because the CSF response model already predicts a
temporal response.

The function :func:`predict_contrast_sensitivity` can be used to predict a contrast sensitivity response to spatial
frequencies (using the aLP CSF).

The function :func:`predict_contrast_response` can be used to predict the contrast response from contrast and contrast
sensitivity (using a Naka-Rushton response function).

References
----------
.. [1] Roelofzen, C., Daghlian, M., van Dijk, J. A., de Jong, M. C., & Dumoulin, S. O. (2025). Modeling neural
    contrast sensitivity functions in human visual cortex. *Imaging Neuroscience*, 3, imag_a_00469.
    https://doi.org/10.1162/imag_a_00469


"""

from ._csf import CSFModel
from ._csf import CSFResponse
from ._csf import predict_contrast_response
from ._csf import predict_contrast_sensitivity

__all__ = [
    "CSFModel",
    "CSFResponse",
    "predict_contrast_response",
    "predict_contrast_sensitivity",
]
