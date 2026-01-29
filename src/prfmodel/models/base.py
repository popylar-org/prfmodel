"""Model base classes."""

from abc import ABC
from abc import abstractmethod
from collections.abc import Sequence
import pandas as pd
from keras import ops
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor


class BatchDimensionError(Exception):
    """
    Exception raised when arguments have different sizes in the batch (first) dimension.

    Parameters
    ----------
    arg_names: Sequence[str]
        Names of arguments that have different sizes in batch dimension.
    arg_shapes: Sequence[tuple of int]
        Shapes of arguments that have different sizes in batch dimension.

    """

    def __init__(self, arg_names: Sequence[str], arg_shapes: Sequence[tuple[int, ...]]):
        names = ", ".join(arg_names)
        shapes = ", ".join([str(s[0]) for s in arg_shapes])

        super().__init__(f"Arguments {names} have different sizes in batch (first) dimension: {shapes}")


class ShapeError(Exception):
    """
    Exception raised when an argument has less than two dimensions.

    Parameters
    ----------
    arg_name: str
        Argument name.
    arg_shape: tuple of int
        Argument shape.

    """

    def __init__(self, arg_name: str, arg_shape: tuple[int, ...]):
        super().__init__(
            f"Argument {arg_name} must have at least two dimensions but has shape {arg_shape}",
        )


class BaseModel(ABC):
    """
    Abstract base class for models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom model classes.
    Subclasses must override the abstract `parameter_names` property.

    Attributes
    ----------
    parameter_names

    Examples
    --------
    Create a custom model class that inherits from the base class:

    >>> class CustomModel(BaseModel):
    >>>     @property
    >>>     def parameter_names(self):
    >>>         return ["a", "b"]
    >>> model = CustomModel()
    >>> print(model.parameter_names)
    ["a", "b"]

    """

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """A list with names of parameters that are used by the model."""


class BasePRFResponse(BaseModel):
    """
    Abstract base class for population receptive field response models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom population receptive field models.
    Subclasses must override the abstract `__call__` method.

    #TODO: Link to Example on how to create custom response models.

    """

    @abstractmethod
    def __call__(self, stimulus: Stimulus, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus.

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, ...)` and dtype `dtype`. The number of voxels is the
            number of rows in `parameters`. The number and size of other axes depends on the stimulus.

        """


class BaseImpulse(BaseModel):
    """
    Abstract base class for impulse response models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom impulse response models.
    Subclasses must override the abstract `__call__` method.

    #TODO: Link to Example on how to create custom impulse response models.

    Parameters
    ----------
    duration : float, default=32.0
        The duration of the impulse response (in seconds).
    offset : float, default=0.0001
        The offset of the impulse response (in seconds). By default a very small offset is added to prevent infinite
        response values at t = 0.
    resolution : float, default=1.0
        The time resultion of the impulse response (in seconds), that is the number of points per second at which the
        impulse response function is evaluated.
    norm : str, optional, default="sum"
        The normalization of the response. Can be `"sum"` (default), `"mean"`, `"max"`, or `None`. If `None`, no
        normalization is performed.
    default_parameters : dict of float, optional
        Dictionary with scalar default parameter values. Keys must be valid parameter names.

    """

    def __init__(
        self,
        duration: float = 32.0,
        offset: float = 0.0001,
        resolution: float = 1.0,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | None = None,
    ):
        super().__init__()

        self.duration = duration
        self.offset = offset
        self.resolution = resolution
        self.norm = norm

        if default_parameters is not None:
            if any(key not in self.parameter_names for key in default_parameters):
                msg = "Invalid default parameter name, please provide valid parameter default parameter names"
                raise ValueError(msg)

            if any(not isinstance(val, float) for val in default_parameters.values()):
                msg = "Default parameters must be single float values"
                raise ValueError(msg)

        self.default_parameters = default_parameters

        self._frames: Tensor | None = None

    @property
    def num_frames(self) -> int:
        """The total number of time frames at which the impulse response function is evaluated."""
        return int(self.duration / self.resolution)

    @property
    def frames(self) -> Tensor:
        """
        The time frames at which the impulse response function is evaluated.

        Time frames are linearly interpolated between `offset` and `duration` and have shape (1, `num_frames`).

        """
        if self._frames is None:
            self._frames = ops.expand_dims(ops.linspace(self.offset, self.duration, self.num_frames), 0)

        return self._frames

    def _join_default_parameters(self, parameters: pd.DataFrame) -> pd.DataFrame:
        if self.default_parameters is not None:
            parameters = parameters.copy()

            for key, val in self.default_parameters.items():
                parameters[key] = val

        return parameters

    @abstractmethod
    def __call__(self, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Compute the impulse response.

        Parameters
        ----------
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, num_frames)` and dtype `dtype`. The number of voxels is the
            number of rows in `parameters`.

        """


class BaseTemporal(BaseModel):
    """
    Abstract base class for temporal models.

    Cannot be instantiated on its own.
    Can only be used as a parent class to create custom temporal models.
    Subclasses must override the abstract `__call__` method.

    #TODO: Link to Example on how to create custom temporal models.

    """

    @abstractmethod
    def __call__(self, inputs: Tensor, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Make predictions with the temporal model.

        Parameters
        ----------
        inputs : Tensor
            Input tensor with temporal response and shape (num_batches, num_frames).
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different batches.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, num_frames)` and dtype `dtype`. The number of voxels is the
            number of rows in `parameters`.

        """


class BasePRFModel(BaseModel):
    """
    Abstract base class for creating composite population receptive field models.

    Cannot be instantiated on its own.
    Can only be used as a parent class for creating custom composite population receptive field models.
    Subclasses must override the `__call__` method.
    This class is intented for combining multiple submodels into a composite model with a custom `__call__`
    method that defines how the submodels interact to make a composite prediction.

    #TODO: Link to Example on how to create custom composite models.

    Parameters
    ----------
    **models
        Submodels to be combined into the composite model. All submodel classes must inherit from `BaseModel`.

    Raises
    ------
    TypeError
        If submodel classes do not inherit from `BaseModel`.

    """

    def __init__(self, **models: BaseModel | None):
        super().__init__()

        for model in models.values():
            if model is not None and not issubclass(model.__class__, BaseModel):
                msg = "Model instance must inherit from BaseModel"
                raise TypeError(msg)

        self.models = models

    @property
    def parameter_names(self) -> list[str]:
        """A list with names of unique parameters that are used by the submodels."""
        param_names = []

        for model in self.models.values():
            if model is not None:
                param_names.extend(model.parameter_names)

        # Make sure no duplicates are returned
        return list(set(param_names))

    @abstractmethod
    def __call__(self, stimulus: Stimulus, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict a composite population receptive field response to a stimulus.

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object.
        parameters : pandas.DataFrame
            Dataframe with columns containing different (sub-) model parameters and rows containing parameter values
            for different voxels.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions with the same shape as `inputs` and dtype `dtype`.
            Model predictions of shape (num_voxels, num_frames). The number of voxels is the number of rows in
            `parameters`. The number of frames is the number of frames in the stimulus design.

        """
