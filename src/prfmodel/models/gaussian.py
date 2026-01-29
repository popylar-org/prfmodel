"""Gaussian population receptive field response models."""

import math
import pandas as pd
from keras import ops
from prfmodel.stimulus import GridDimensionsError
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import _MIN_PARAMETER_DIM
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype
from .base import BaseImpulse
from .base import BasePRFResponse
from .base import BaseTemporal
from .base import BatchDimensionError
from .base import ShapeError
from .composite import SimplePRFModel
from .impulse import ShiftedDerivativeGammaImpulse
from .temporal import BaselineAmplitude


class GridMuDimensionsError(Exception):
    """
    Exception raised when the dimensions of the stimulus grid and the Gaussian mu parameter do not match.

    For a stimulus grid with shape (..., m), the shape of the Gaussian mu parameter must be (num_batches, m).

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the stimulus grid.
    mu_shape : tuple of int
        Shape of the Gaussian mu parameter.

    """

    def __init__(self, grid_shape: tuple[int, ...], mu_shape: tuple[int, ...]):
        super().__init__(f"For 'grid' {grid_shape} and 'mu' {mu_shape} do not match")


def _check_gaussian_args(grid: Tensor, mu: Tensor, sigma: Tensor) -> None:
    if not len(grid.shape[:-1]) == grid.shape[-1]:
        raise GridDimensionsError(grid.shape)

    if len(mu.shape) < _MIN_PARAMETER_DIM:
        raise ShapeError(
            arg_name="mu",
            arg_shape=mu.shape,
        )

    if len(sigma.shape) < _MIN_PARAMETER_DIM:
        raise ShapeError(
            arg_name="sigma",
            arg_shape=sigma.shape,
        )

    if grid.shape[-1] != mu.shape[-1]:
        raise GridMuDimensionsError(grid.shape, mu.shape)

    if mu.shape[0] != sigma.shape[0]:
        raise BatchDimensionError(
            arg_names=("mu", "sigma"),
            arg_shapes=(mu.shape, sigma.shape),
        )


def _expand_gaussian_args(grid: Tensor, mu: Tensor, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # Expand mu to same shape as grid: (num_batches, ..., grid.shape[-1])
    mu_expand = tuple(range(1, grid.shape[-1] + 1))
    # Expand sigma to same shape as grid but omit last grid dimension
    sigma_expand = mu_expand[:-1]

    mu = ops.expand_dims(mu, axis=mu_expand)
    sigma = ops.expand_dims(sigma, axis=sigma_expand)
    # Add new first dimension to grid corresponding to num_batches
    grid = ops.expand_dims(grid, axis=0)

    return grid, mu, sigma


def predict_gaussian_response(grid: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """
    Predict a isotropic Gaussian population receptive field response.

    The dimensionality of the Gaussian depends on the number of dimensions of `grid` and `mu`. All dimensions have
    the same size `sigma`.

    Parameters
    ----------
    grid : Tensor
        Stimulus grid for which to make predictions.
    mu : Tensor
        Centroid of the population receptive field. Must have at least two dimensions.
        The first dimension corresponds to the number of batches.
        The second dimension corresponds to the number of grid dimensions and must match the size of the
        last `grid` dimension.
    sigma : Tensor
        Size of the population receptive field. Must have at least two dimensions.
        The first dimension corresponds to the number of batches,
        and its size must match the size of the first `mu` dimension.
        The second dimension must have size one (because all Gaussian dimensions have the same size).

    Returns
    -------
    Tensor
        The predicted Gaussian population receptive field response with shape (num_batches, ...)
        where `...` corresponds to the dimensions of the Gaussian.

    Raises
    ------
    BatchDimensionError
        If `mu` and `sigma` have batch (first) dimensions with different sizes.
    GridDimensionsError
        If the grid has mismatching dimensions.
    GridMuDimensionsError
        If the grid and mu dimensions do not match.
    ParameterShapeError
        If `mu` or `sigma` have less than two dimensions.

    Examples
    --------
    Predict a 2D Gaussian response:

    >>> import numpy as np
    >>> # Define a 2D grid
    >>> num_x, num_y = 10, 10
    >>> x = np.linspace(-3, 3, num_x)
    >>> y = np.linspace(-4, 4, num_y)
    >>> xv, yv = np.meshgrid(x, y)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (10, 10, 2)
    >>> # Define 2D centroids of Gaussian for 3 batches
    >>> mu = np.array([ # shape (3, 2), first column y, second column x
    >>>     [0.0, 1.0],
    >>>     [1.0, 0.0],
    >>>     [0.0, 0.0],
    >>> ])
    >>> # Define size of Gaussian for 3 batches
    >>> sigma = np.array([[1.0], [1.5], [2.0]]) # shape (3, 1)
    >>> resp = predict_gaussian_response(grid, mu, sigma)
    >>> print(resp.shape) # (num_batches, num_y, num_x)
    (3, 10, 10)

    """
    grid = ops.convert_to_tensor(grid)
    mu = ops.convert_to_tensor(mu)
    sigma = ops.convert_to_tensor(sigma)

    num_dims = grid.shape[-1]

    _check_gaussian_args(grid, mu, sigma)

    # Expand axes to enable keras.ops autocasting
    grid, mu, sigma = _expand_gaussian_args(grid, mu, sigma)

    sigma_squared = ops.square(sigma)

    # Gaussian response
    resp = ops.sum(ops.square(grid - mu), axis=-1)
    resp /= 2 * sigma_squared

    # Divide by volume to normalize
    volume = (2 * math.pi * sigma_squared) ** (num_dims / 2)

    return ops.exp(-resp) / volume


class Gaussian2DResponse(BasePRFResponse):
    """
    Two-dimensional isotropic Gaussian population receptive field response model.

    Predicts a response to a stimulus grid.
    The model has three parameters: `mu_y` and `mu_x` for the center and `sigma` for the width of the Gaussian.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import prfmodel as pm
    >>> # Define a 2D grid
    >>> num_x, num_y = 20, 10
    >>> x = np.linspace(-3, 3, num_x)
    >>> y = np.linspace(-4, 4, num_y)
    >>> xv, yv = np.meshgrid(x, y)
    >>> grid = np.stack((xv, yv), axis=-1)  # shape (20, 10, 2)
    >>> # Define 2D centroids of Gaussian for 3 voxels
    >>> params = pd.DataFrame({
    >>>     "mu_x": [0.0, 1.0, 0.0],
    >>>     "mu_y": [1.0, 0.0, 0.0],
    >>>     "sigma": [1.0, 1.5, 2.0],
    >>> })
    >>> # Define dummy design for 10 frames
    >>> design = np.ones(10, num_y, num_x)
    >>> # Create stimulus object
    >>> stimulus = pm.Stimulus(
    >>>     design=design,
    >>>     grid=grid,
    >>>     dimension_labels=("y", "x"),
    >>> )
    >>> # Create model instance
    >>> model = Gaussian2DResponse()
    >>> # Predict response to stimulus grid
    >>> resp = model(stimulus, params)
    >>> print(resp.shape) # (num_voxels, num_y, num_x)
    (3, 20, 10)
    """

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `mu_y`, `mu_x`, `sigma`."""
        return ["mu_y", "mu_x", "sigma"]

    def __call__(self, stimulus: Stimulus, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus with a 2D grid.

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object with a 2D stimulus grid.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels. Must contain the columns `mu_y`, `mu_x` and `sigma`.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, size_y, size_x)` and dtype `dtype`.
            `num_voxels` is the number of rows in `parameters` and `size_y` and `size_x` are the sizes of the
            x and y stimulus grid dimension.

        """
        dtype = get_dtype(dtype)
        # Convention is y-dimension first
        mu = convert_parameters_to_tensor(parameters[["mu_y", "mu_x"]], dtype=dtype)
        sigma = convert_parameters_to_tensor(parameters[["sigma"]], dtype=dtype)
        grid = ops.convert_to_tensor(stimulus.grid, dtype=dtype)

        return predict_gaussian_response(grid, mu, sigma)


class Gaussian2DPRFModel(SimplePRFModel):
    """
    Two-dimensional isotropic Gaussian population receptive field model.

    This is a generic class that combines a 2D isotropic Gaussian population receptive field, impulse,
    and temporal model response.

    Parameters
    ----------
    impulse_model : BaseImpulse or type or None, default=ShiftedDerivativeGammaImpulse, optional
        An impulse response model class or instance. Reponse model classes will be instantiated during object
        initialization. The default creates a :class:`~prfmodel.models.impulse.ShiftedDerivativeGammaImpulse`
        instance with default values.
    temporal_model : BaseTemporal or type or None, default=BaselineAmplitude, optional
        A temporal model class or instance. Temporal model instances will be instantiated during initialization.
        The default creates a `BaselineAmplitude` instance.

    Notes
    -----
    The simple composite model follows five steps:

    1. The 2D Gaussian population receptive field response model makes a prediction for the stimulus grid.
    2. The response is encoded with the stimulus design.
    3. A impulse response model generates an impulse response.
    4. The encoded response is convolved with the impulse response.
    5. The temporal model modifies the convolved response.

    """

    def __init__(
        self,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = ShiftedDerivativeGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = BaselineAmplitude,
    ):
        super().__init__(
            prf_model=Gaussian2DResponse(),
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )
