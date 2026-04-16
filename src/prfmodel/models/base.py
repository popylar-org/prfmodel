"""Generic abstract base classes for response, stimulus encoder, and canonical models.

Classes in this module inherit from :class:`~prfmodel.utils.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.utils.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example :class:`~prfmodel.models.base.BaseResponse` defines that all child classes
must implement a :meth:`~prfmodel.models.base.BaseResponse.__call__` method that takes a stimulus and set of parameters
as input. However, it leaves it up to each child class to define how input stimulus and parameters are used to make
model predictions.

Classes in this module are also generic with respect to the input stimulus, that is, child classes can specify whether
they take a :class:`~prfmodel.stimuli.PRFStimulus` or :class:`~prfmodel.stimuli.CFStimulus` as input. In the case of
:meth:`~prfmodel.models.base.BaseResponse`, child classes can choose the type of input stimulus in the signature of
:meth:`~prfmodel.models.base.BaseResponse.__call__`.

"""

from abc import abstractmethod
from typing import Generic
from typing import TypeVar
import pandas as pd
from prfmodel._docstring import doc
from prfmodel.stimuli import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import ModelProtocol

S = TypeVar("S", bound=Stimulus)


class BaseResponse(ModelProtocol, Generic[S]):
    """
    Generic abstract base class for response models.

    A response model takes a stimulus and parameters as input and predicts a response.

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom response
    models. Subclasses must override the abstract :attr:`parameter_names` and :meth:`__call__` method.
    They must be defined with a specific stimulus type. See :mod:`~prfmodel.models.base` for details.

    Examples
    --------
    Reimplement a 2D isotropic Gaussian response model for a :class:`~prfmodel.stimuli.PRFStimulus`.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.stimuli import PRFStimulus
    >>> from prfmodel.models.prf import predict_gaussian_response
    >>> from prfmodel.utils import convert_parameters_to_tensor, get_dtype
    >>> from keras import ops
    >>> # Define custom child class
    >>> class CustomGaussian2DResponse(BaseResponse[PRFStimulus]):
    ...     @property
    ...     def parameter_names(self):
    ...         return ["mu_y", "mu_x", "sigma"]
    ...     def __call__(self, stimulus, parameters, dtype=None):
    ...         dtype = get_dtype(dtype)
    ...         mu = convert_parameters_to_tensor(parameters[["mu_y", "mu_x"]], dtype=dtype)
    ...         sigma = convert_parameters_to_tensor(parameters[["sigma"]], dtype=dtype)
    ...         grid = ops.convert_to_tensor(stimulus.grid, dtype=dtype)
    ...         return predict_gaussian_response(grid, mu, sigma)
    >>> # Load example pRF stimulus
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> # Define parameters
    >>> params = pd.DataFrame({
    ...     "mu_y": [0.0, 1.0],
    ...     "mu_x": [1.0, 0.0],
    ...     "sigma": [1.0, 1.5],
    ... })
    >>> # Create child model instance
    >>> model = CustomGaussian2DResponse()
    >>> # Make model prediction for example stimulus
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_y, num_x)
    (2, 101, 101)

    """

    @doc
    @abstractmethod
    def __call__(self, stimulus: S, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus.

        Parameters
        ----------
        %(stimulus)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Model predictions of shape `(num_units, ...)` and dtype `dtype`. The number of units is the
            number of rows in `parameters`. The number and size of other axes depends on the stimulus.

        """


class BaseEncoder(ModelProtocol, Generic[S]):
    """
    Generic abstract base class for encoding model responses with a stimulus.

    A stimulus encoder takes a model response and stimulus as input and predicts a stimulus-encoded model response.

    Notes
    -----
    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom stimulus encoding models.
    Subclasses must override the abstract :attr:`parameter_names` property and
    :meth:`__call__` method and must be defined with a specific stimulus type.
    See :mod:`~prfmodel.models.base` for details.

    Examples
    --------
    Reimplement a stimulus encoder for a :class:`~prfmodel.stimuli.PRFStimulus` that encodes the response
    by multiplying with the stimulus design and summing over the spatial dimensions.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.stimuli import PRFStimulus
    >>> from prfmodel.models.prf import encode_prf_response
    >>> from prfmodel.utils import get_dtype
    >>> from keras import ops
    >>> class CustomPRFEncoder(BaseEncoder[PRFStimulus]):
    ...     @property
    ...     def parameter_names(self):
    ...         return []
    ...     def __call__(self, stimulus, response, parameters, dtype=None):
    ...         dtype = get_dtype(dtype)
    ...         design = ops.convert_to_tensor(stimulus.design, dtype=dtype)
    ...         return encode_prf_response(response, design, dtype=dtype)
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> response = np.ones((3, 101, 101))  # dummy response of shape (num_units, num_y, num_x)
    >>> params = pd.DataFrame()
    >>> encoder = CustomPRFEncoder()
    >>> encoded = encoder(stimulus, response, params)
    >>> print(encoded.shape)  # (num_units, num_frames)
    (3, 200)

    """

    @doc
    @abstractmethod
    def __call__(
        self,
        stimulus: S,
        response: Tensor,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ):
        """Encode a model response with a stimulus.

        Parameters
        ----------
        %(stimulus)s
        response : :data:`prfmodel.typing.Tensor`
            Model response.
        %(parameters)s
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The stimulus encoded model response with shape `(num_units, ...)` dtype `dtype`. The number of units is
            the number of rows in :attr:`parameters`. The number and size of other axes depends on the stimulus and the
            response.

        """


class BaseCanonical(ModelProtocol, Generic[S]):
    """
    Generic abstract base class for creating canonical models.

    A canonical model combines multiple submodels and defines how they interact to make a combined prediction.

    Parameters
    ----------
    **models
        Submodels to be combined into the canonical model. All submodel classes must inherit from
        :class:`~prfmodel.utils.ModelProtocol`.

    Raises
    ------
    TypeError
        If submodel classes do not inherit from :class:`~prfmodel.utils.ModelProtocol`.

    Notes
    -----
    Cannot be instantiated on its own. Can only be used as a parent class to create custom canonical models.
    Subclasses must override the abstract :meth:`__call__` method and must be defined
    with a specific stimulus type.

    Examples
    --------
    Create a canonical model that combines a :class:`~prfmodel.models.prf.Gaussian2DPRFResponse` and a
    :class:`~prfmodel.models.prf.PRFStimulusEncoder`. The :attr:`parameter_names` property automatically
    aggregates the unique parameter names from all submodels.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.stimuli import PRFStimulus
    >>> from prfmodel.models.prf import Gaussian2DPRFResponse, PRFStimulusEncoder
    >>> class SimplePRFComposite(BaseCanonical[PRFStimulus]):
    ...     def __call__(self, stimulus, parameters, dtype=None):
    ...         response = self.models["prf_response"](stimulus, parameters, dtype=dtype)
    ...         return self.models["encoder"](stimulus, response, parameters, dtype=dtype)
    >>> model = SimplePRFComposite(
    ...     prf_response=Gaussian2DPRFResponse(),
    ...     encoder=PRFStimulusEncoder(),
    ... )
    >>> model.parameter_names
    ['mu_y', 'mu_x', 'sigma']
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> params = pd.DataFrame({"mu_y": [0.0, 1.0], "mu_x": [1.0, 0.0], "sigma": [1.0, 1.5]})
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (2, 200)

    """

    def __init__(self, **models: ModelProtocol | None):
        super().__init__()

        for model in models.values():
            if model is not None and not isinstance(model, ModelProtocol):
                msg = "Model instance must implement the 'parameter_names' property"
                raise TypeError(msg)

        self.models = models

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters that are used by the submodels."""
        param_names = []

        for model in self.models.values():
            if model is not None:
                param_names.extend(model.parameter_names)

        # Make sure no duplicates are returned (preserve insertion order)
        return list(dict.fromkeys(param_names))

    @doc
    @abstractmethod
    def __call__(
        self,
        stimulus: S,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a canonical model response to a stimulus.

        Parameters
        ----------
        %(stimulus)s
        %(parameters)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        """
