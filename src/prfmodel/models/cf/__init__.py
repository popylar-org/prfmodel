"""Connective field (CF) models.

This module contains CF response models that predict a CF response to a stimulus source distance matrix.
(e.g., :class:`GaussianCFResponse`). Because the source distance matrix is always two-dimensional
in the canonical CF model formulation, CF response models always operate on two spatial dimensions.

The module also contains the
:class:`CFStimulusEncoder` class that encodes a CF response with a stimulus source response.

It also contains a canonical model that combines a CF response with a stimulus encoding model and a scaling model
from :mod:`~prfmodel.scaling`. In contrast to pRF models, canonical CF models do not require an impulse response
model because the source response already exhibits the physiological shape of the target neural signal.

Currently, only the canonical Gaussian CF model (:class:`GaussianCFModel`) [1]_ is implemented but it is possible to
implement more complex models analogously to those implemented in the :mod:`~prfmodel.models.prf` module.

References
----------
.. [1] Haak, K. V., Winawer, J., Harvey, B. M., Renken, R., Dumoulin, S. O., Wandell, B. A., & Cornelissen, F. W.
    (2013). Connective field modeling. *NeuroImage*, *66*, 376-384. https://doi.org/10.1016/j.neuroimage.2012.10.037

"""

from ._gaussian import GaussianCFModel
from ._gaussian import GaussianCFResponse
from ._stimulus_encoding import CFStimulusEncoder

__all__ = [
    "CFStimulusEncoder",
    "GaussianCFModel",
    "GaussianCFResponse",
]
