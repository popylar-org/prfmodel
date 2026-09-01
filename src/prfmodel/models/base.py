"""Generic abstract base classes for response, stimulus encoder, and canonical models.

Classes in this module inherit from :class:`~prfmodel.protocols.ModelProtocol` that requires them to implement a
:attr:`~prfmodel.protocols.ModelProtocol.parameter_names` property.

They are abstract base classes, meaning that they
cannot be instantiated on their own but are intended as parent classes that define attributes and methods that are
shared by all child classes. For example, :class:`~prfmodel.models.base.BasePopulationResponse` defines that all child
classes must implement a :meth:`~prfmodel.models.base.BasePopulationResponse.call` method that takes a tensor-holding
stimulus and set of tensor parameters as input. However, it leaves it up to each child class to define how input
stimulus and parameters are used to make model predictions.

All base classes have a concrete user-facing :meth:``__call__`` method
(e.g., :meth:`~prfmodel.models.base.BasePopulationResponse.__call__`) that takes non-tensor arguments and
performs validation checks. This method calls the abstract ``call`` method that must be implemented by each child
class and only accepts tensor arguments to enable backend compilation.

Validation checks that read a value back to a Python `bool` are done in `__call__` only, because `call` may be traced.

Classes in this module are also generic with respect to the input stimulus. This means that child classes must specify
which user-facing and tensor-holding stimulus types :meth:`__call__` and :meth:`call` take as input.

An exception is :class:`~prfmodel.models.base.BaseCanonical` which is a composite model class that is intended for
holding and calling submodels that inherit from :class:`~prfmodel.protocols.ModelProtocol`. Child classes must only
define the :meth:`call` method and optionally the :attr`_additional_parameter_names` attribute. The composite model
class collects the parameter names of all submodels and performs validation checks on them.

"""

from abc import abstractmethod
from typing import Generic
from typing import TypeVar
from typing import cast
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.protocols import CompositeModelProtocol
from prfmodel.protocols import ModelProtocol
from prfmodel.regressors.base import BaseRegressors
from prfmodel.regressors.base import _extract_regressor_design
from prfmodel.regressors.base import _validate_regressors_argument
from prfmodel.stimuli import Stimulus
from prfmodel.stimuli import StimulusTensors
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import as_tensor_frame
from prfmodel.utils import get_dtype

S = TypeVar("S", bound=Stimulus)
"""User-facing stimulus type, e.g. :class:`~prfmodel.stimuli.PRFStimulus`."""

T = TypeVar("T", bound=StimulusTensors)
"""Tensor-holding stimulus type, e.g. :class:`~prfmodel.stimuli.PRFStimulusTensors`.

'S.to_tensors()' always returns the matching 'T', e.g., 'PRFStimulus.to_tensors' returns a
'PRFStimulusTensors'.
"""


class BasePopulationResponse(ModelProtocol, Generic[S, T]):
    """
    Generic abstract base class for neuron population response models.

    A neuron population response model takes a stimulus and parameters as input and predicts a population response.

    Notes
    -----
    This class cannot be instantiated on its own. It can only be used as a parent class to create custom response
    models. Subclasses must override the abstract :attr:`parameter_names` property and the :meth:`call` method,
    and must be defined with a specific user-facing stimulus type and its matching tensor-holding type.
    See :mod:`~prfmodel.models.base` for details.

    Do not override :meth:`__call__`. It is the public facade that validates the parameters, resolves the dtype
    and converts both the stimulus and the parameters to tensors before handing them to :meth:`call`.

    Examples
    --------
    Reimplement a 2D isotropic Gaussian response model for a :class:`~prfmodel.stimuli.PRFStimulus`.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.stimuli import PRFStimulus, PRFStimulusTensors
    >>> from prfmodel.models.prf import predict_gaussian_response
    >>> # Define custom child class
    >>> class CustomGaussian2DResponse(BasePopulationResponse[PRFStimulus, PRFStimulusTensors]):
    ...     @property
    ...     def parameter_names(self):
    ...         return ["mu_y", "mu_x", "sigma"]
    ...     def call(self, stimulus, parameters):
    ...         return predict_gaussian_response(
    ...             stimulus.grid, parameters[["mu_y", "mu_x"]], parameters[["sigma"]]
    ...         )
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
    (2, 128, 128)

    """

    @doc
    def __call__(self, stimulus: S, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus.

        This is the public entry point. It accepts the user-facing types, validates them, converts them to
        tensors and delegates the arithmetic to :meth:`call`. Subclasses implement :meth:`call`, not this
        method.

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

        Raises
        ------
        %(raises_missing_parameters)s

        """
        dtype = get_dtype(dtype)
        self.check_parameter_names(parameters)
        self.check_parameter_values(parameters)

        return self.call(
            cast("T", stimulus.to_tensors(dtype)),
            as_tensor_frame(parameters[self.parameter_names], dtype),
        )

    @doc
    @abstractmethod
    def call(self, stimulus: T, parameters: TensorFrame) -> Tensor:
        """
        Predict the model response from tensors.

        Parameters
        ----------
        %(stimulus_tensors)s
        %(parameters_tensors)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            Model predictions of shape `(num_units, ...)`.

        Notes
        -----
        Implementations must be traceable by a backend compiler, because this is the method the fitters
        wrap in ``tf.function`` or ``jax.jit``. In practice: use :mod:`keras.ops` only, never :mod:`numpy`
        or :mod:`pandas`, and never branch on a tensor *value*. Branching on a tensor *shape* is fine, since shapes are
        known at trace time. Checks that need concrete values belong in :meth:`__call__`.

        """


class BaseStimulusEncoder(ModelProtocol, Generic[S, T]):
    """
    Generic abstract base class for encoding model responses with a stimulus.

    A stimulus encoder takes a model response and stimulus as input and predicts a stimulus-encoded model response.

    Notes
    -----
    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom stimulus encoding models.
    Subclasses must override the abstract :attr:`parameter_names` property and the :meth:`call` method, and
    must be defined with a specific stimulus type and its matching tensor-holding type.
    See :mod:`~prfmodel.models.base` for details.

    Do not override :meth:`__call__`; it is the public facade that validates and converts before delegating
    to :meth:`call`.

    Examples
    --------
    Reimplement a stimulus encoder for a :class:`~prfmodel.stimuli.PRFStimulus` that encodes the response
    by multiplying with the stimulus design and summing over the spatial dimensions.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.stimuli import PRFStimulus, PRFStimulusTensors
    >>> from prfmodel.models.prf import encode_prf_response
    >>> class CustomPRFStimulusEncoder(BaseStimulusEncoder[PRFStimulus, PRFStimulusTensors]):
    ...     @property
    ...     def parameter_names(self):
    ...         return []
    ...     def call(self, stimulus, response, parameters):
    ...         return encode_prf_response(response, stimulus.design, dtype=parameters.dtype)
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> response = np.ones((3, 128, 128))  # dummy response of shape (num_units, num_y, num_x)
    >>> params = pd.DataFrame()
    >>> encoder = CustomPRFStimulusEncoder()
    >>> encoded = encoder(stimulus, response, params)
    >>> print(encoded.shape)  # (num_units, num_frames)
    (3, 170)

    """

    @doc
    def __call__(
        self,
        stimulus: S,
        response: Tensor,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """Encode a model response with a stimulus.

        This is the public entry point; subclasses implement :meth:`call` instead.

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

        Raises
        ------
        %(raises_missing_parameters)s

        """
        dtype = get_dtype(dtype)
        self.check_parameter_names(parameters)
        self.check_parameter_values(parameters)

        return self.call(
            cast("T", stimulus.to_tensors(dtype)),
            ops.convert_to_tensor(response, dtype=dtype),
            as_tensor_frame(parameters[self.parameter_names], dtype),
        )

    @doc
    @abstractmethod
    def call(self, stimulus: T, response: Tensor, parameters: TensorFrame) -> Tensor:
        """Encode a model response with a stimulus, from tensors.

        Parameters
        ----------
        %(stimulus_tensors)s
        response : :data:`prfmodel.typing.Tensor`
            Model response.
        %(parameters_tensors)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The stimulus encoded model response with shape `(num_units, ...)`.

        Notes
        -----
        Implementations must be traceable by a backend compiler. See
        :meth:`BasePopulationResponse.call` for details.

        """


class BaseCanonical(CompositeModelProtocol, Generic[S, T]):
    """
    Generic abstract base class for creating canonical models.

    A canonical model combines multiple submodels and defines how they interact to make a combined prediction.

    Parameters
    ----------
    **models
        Submodels to be combined into the canonical model. All submodel classes must inherit from
        :class:`~prfmodel.protocols.ModelProtocol`.

    Raises
    ------
    TypeError
        If submodel classes do not inherit from :class:`~prfmodel.protocols.ModelProtocol`.

    Notes
    -----
    Cannot be instantiated on its own. Can only be used as a parent class to create custom canonical models.
    Subclasses must override the abstract :meth:`call` method and must be defined with a specific stimulus type
    and its matching tensor-holding type. Do not override :meth:`__call__`.

    Inside :meth:`call`, invoke submodels through *their* :meth:`call` as well, not through the user-facing
    :meth:`__call__`.

    Examples
    --------
    Create a canonical model that combines a :class:`~prfmodel.models.prf.Gaussian2DPRFResponse` and a
    :class:`~prfmodel.models.prf.PRFStimulusEncoder`. The :attr:`parameter_names` property automatically
    aggregates the unique parameter names from all submodels.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.stimuli import PRFStimulus, PRFStimulusTensors
    >>> from prfmodel.models.prf import Gaussian2DPRFResponse, PRFStimulusEncoder
    >>> class CanonicalPRFModel(BaseCanonical[PRFStimulus, PRFStimulusTensors]):
    ...     def call(self, stimulus, parameters, regressors=None):
    ...         response = self.models["prf_model"].call(stimulus, parameters)
    ...         return self.models["encoding_model"].call(stimulus, response, parameters)
    >>> model = CanonicalPRFModel(
    ...     prf_model=Gaussian2DPRFResponse(),
    ...     encoding_model=PRFStimulusEncoder(),
    ... )
    >>> model.parameter_names
    ['mu_y', 'mu_x', 'sigma']
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> params = pd.DataFrame({"mu_y": [0.0, 1.0], "mu_x": [1.0, 0.0], "sigma": [1.0, 1.5]})
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (2, 170)

    """

    def __init__(self, **models: ModelProtocol | None):
        super().__init__()

        if models:
            self.models = models

    @doc
    def __call__(
        self,
        stimulus: S,
        parameters: pd.DataFrame,
        regressors: pd.DataFrame | None = None,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict a canonical model response to a stimulus.

        This is the public entry point; subclasses implement :meth:`call` instead.

        Parameters
        ----------
        %(stimulus)s
        %(parameters)s
        %(regressors_canonical)s
        %(dtype)s

        Returns
        -------
        %(predicted_response_2d)s

        Raises
        ------
        %(raises_missing_parameters)s

        """
        dtype = get_dtype(dtype)
        regressors_model = cast("BaseRegressors | None", self.models.get("regressors_model"))

        self.check_parameter_names(parameters)
        self.check_parameter_values(parameters)
        _validate_regressors_argument(regressors_model, regressors)

        return self.call(
            cast("T", stimulus.to_tensors(dtype)),
            as_tensor_frame(parameters[self.get_consumed_parameter_names(parameters)], dtype),
            _extract_regressor_design(regressors_model, regressors, dtype),
        )

    @doc
    @abstractmethod
    def call(self, stimulus: T, parameters: TensorFrame, regressors: TensorFrame | None = None) -> Tensor:
        """
        Predict a canonical model response from tensors.

        Parameters
        ----------
        %(stimulus_tensors)s
        %(parameters_tensors)s
        %(regressors_tensors)s

        Returns
        -------
        %(predicted_response_2d)s

        Notes
        -----
        Implementations must be traceable by a backend compiler, and must reach submodels through their
        :meth:`call` rather than through :meth:`__call__`. See :meth:`BasePopulationResponse.call`.

        """
