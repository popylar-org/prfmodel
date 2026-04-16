"""Neural response models.

This module contains models that map neural responses to input stimuli. Currently, it implements population receptive
field (pRF) and connective field (CF) models.

All models in in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:meth:`~prfmodel.utils.ModelProtocol.parameter_names` property.

The :mod:`~prfmodel.models.base` submodule contains generic abstract base classes that pRF and CF models inherit from.

"""
