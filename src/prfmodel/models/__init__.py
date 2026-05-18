"""Neural response models.

This module contains models that map neural responses to input stimuli. These typically contain parameters of
interest. Currently, it implements population receptive field (pRF) and connective field (CF) models.

The :mod:`~prfmodel.models.base` submodule contains generic abstract base classes that pRF and CF models inherit from.

The :mod:`~prfmodel.models.compression` submodule contains generic classes to (de-) compress stimulus-encoded model
responses.

Notes
-----
All models in in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

"""
