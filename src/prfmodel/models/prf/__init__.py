"""Population receptive field (pRF) models.

This module contains pRF response models that predict a pRF response to a stimulus grid
(e.g., :class:`Gaussian2DPRFResponse`).

It also contains the
:class:`PRFStimulusEncoder` class that encodes a pRF response with a stimulus design.

Finally, it
contains multiple canonical models that combine a pRF response with a stimulus encoding model, an impulse response
model from :mod:`~prfmodel.impulse`, and a scaling model from :mod:`~prfmodel.scaling`. Currently, canonical Gaussian
[1]_, Compressive Spatial Summation (CSS) [2]_, Difference of Gaussian (DoG) [3]_, and
Divisive Normalization (DivNorm; DN) [4]_ models are implemented. These models increase in complexity, with the
canonical Gaussian being the least and the DN model being the most complex model.

There are also multiple helper functions that initialize starting parameters from parameter estimates obtained from a
different model (e.g., :func:`init_css_from_gaussian`).

The function :func:`encode_prf_response` can be used to stimulus-encode a pRF response with arbitray dimensions. This
function is also internally used by :class:`PRFStimulusEncoder`.

The function :func:`predict_gaussian_response` can be used to make Gaussian density predictions to stimulus grids with
arbitrary dimensions. This function is also used internally by :class:`Gaussian2DPRFResponse`.

References
----------
.. [1] Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex.
    *NeuroImage*, *39*(2), 647-660. https://doi.org/10.1016/j.neuroimage.2007.09.034
.. [2] Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual
    cortex. *Journal of Neurophysiology*, *110*(2), 481-494. https://doi.org/10.1152/jn.00105.2013
.. [3] Zuiderbaan, W., Harvey, B. M., & Dumoulin, S. O. (2012). Modeling center-surround configurations in population
    receptive fields using fMRI. *Journal of Vision*, *12*(3), 10. https://doi.org/10.1167/12.3.10
.. [4] Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures
    throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118.
    https://doi.org/10.1073/pnas.2108713118

"""

from ._css import Gaussian2DCSSPRFModel
from ._css import init_css_from_gaussian
from ._div_norm import DivNormGaussian2DPRFModel
from ._div_norm import DivNormPRFModel
from ._div_norm import init_dn_from_dog
from ._div_norm import init_dn_from_gaussian
from ._dog import DoG2DPRFModel
from ._dog import init_dog_from_gaussian
from ._gaussian import Gaussian2DPRFModel
from ._gaussian import Gaussian2DPRFResponse
from ._gaussian import predict_gaussian_response
from ._stimulus_encoding import PRFStimulusEncoder
from ._stimulus_encoding import encode_prf_response

__all__ = [
    "DivNormGaussian2DPRFModel",
    "DivNormPRFModel",
    "DoG2DPRFModel",
    "Gaussian2DCSSPRFModel",
    "Gaussian2DPRFModel",
    "Gaussian2DPRFResponse",
    "PRFStimulusEncoder",
    "encode_prf_response",
    "init_css_from_gaussian",
    "init_dn_from_dog",
    "init_dn_from_gaussian",
    "init_dog_from_gaussian",
    "predict_gaussian_response",
]
