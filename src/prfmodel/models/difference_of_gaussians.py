import pandas as pd
from keras import ops
from prfmodel.models.base import BasePRFResponse
from prfmodel.models.gaussian import predict_gaussian_response
from prfmodel.stimulus import Stimulus
from prfmodel.typing import Tensor
from prfmodel.utils import convert_parameters_to_tensor
from prfmodel.utils import get_dtype


class DifferenceOfGaussians2DResponse(BasePRFResponse):
    """
    Two-dimensional difference of Gaussians population receptive field response model.

    Predicts a response to a stimulus grid.
    The model has four parameters:
        - `mu_y` and `mu_x` for the center and `sigma1` for the width of the first Gaussian.
        - `sigma2` for the width of the second Gaussian (shares the center w/ the first Gaussian: `mu_y` and `mu_x`).

    The response is computed as Gaussian(sigma1) - Gaussian(sigma2). ``sigma2`` must be greater than or equal to
    ``sigma1`` for all voxels.

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
    >>> # Define parameters for 3 voxels
    >>> params = pd.DataFrame({
    >>>     "mu_x": [0.0, 1.0, 0.0],
    >>>     "mu_y": [1.0, 0.0, 0.0],
    >>>     "sigma1": [1.0, 1.5, 2.0],
    >>>     "sigma2": [2.0, 3.0, 4.0],
    >>> })
    >>> # Define dummy design for 10 frames
    >>> design = np.ones((10, num_y, num_x))
    >>> # Create stimulus object
    >>> stimulus = pm.Stimulus(
    >>>     design=design,
    >>>     grid=grid,
    >>>     dimension_labels=("y", "x"),
    >>> )
    >>> # Create model instance
    >>> model = DifferenceOfGaussians2DResponse()
    >>> # Predict response to stimulus grid
    >>> resp = model(stimulus, params)
    >>> print(resp.shape) # (num_voxels, num_y, num_x)
    (3, 20, 10)
    """

    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `mu_y`, `mu_x`, `sigma1`, `sigma2`."""
        return ["mu_y", "mu_x", "sigma1", "sigma2"]

    def __call__(self, stimulus: Stimulus, parameters: pd.DataFrame, dtype: str | None = None) -> Tensor:
        """
        Predict the model response for a stimulus with a 2D grid.

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object with a 2D stimulus grid.
        parameters : pandas.DataFrame
            Dataframe with columns containing different model parameters and rows containing parameter values
            for different voxels. Must contain the columns `mu_y`, `mu_x`, `sigma1` and `sigma2`.
            ``sigma2`` must be greater than or equal to ``sigma1`` for all voxels.
        dtype : str, optional
            The dtype of the prediction result. If `None` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_voxels, size_y, size_x)` and dtype `dtype`.
            `num_voxels` is the number of rows in `parameters` and `size_y` and `size_x` are the sizes of the
            x and y stimulus grid dimension.

        Raises
        ------
        ValueError
            If any voxel has ``sigma2 < sigma1``.

        """
        if (parameters["sigma2"] < parameters["sigma1"]).any():
            msg = "sigma2 must be greater than or equal to sigma1 for all voxels"
            raise ValueError(msg)

        dtype = get_dtype(dtype)
        mu = convert_parameters_to_tensor(parameters[["mu_y", "mu_x"]], dtype=dtype)
        sigma1 = convert_parameters_to_tensor(parameters[["sigma1"]], dtype=dtype)
        sigma2 = convert_parameters_to_tensor(parameters[["sigma2"]], dtype=dtype)
        grid = ops.convert_to_tensor(stimulus.grid, dtype=dtype)

        resp1 = predict_gaussian_response(grid, mu, sigma1)
        resp2 = predict_gaussian_response(grid, mu, sigma2)

        return resp1 - resp2
