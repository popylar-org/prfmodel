"""Library of canonical hemodynamic impulse responses."""

from functools import cache
from importlib.resources import files
import numpy as np
import pandas as pd
from keras import ops
from prfmodel._docstring import doc
from prfmodel.typing import Tensor
from prfmodel.utils import TensorFrame
from prfmodel.utils import get_dtype
from prfmodel.utils import normalize_response
from .base import BaseImpulse

HRF_LIBRARY_RESOLUTION = 0.1
"""Time resolution (in seconds) at which the HRFs in the GLMsingle library are sampled."""

HRF_LIBRARY_SIZE = 20
"""Number of HRFs in the GLMsingle library."""


@cache
def _load_hrf_library() -> np.ndarray:
    """Load the GLMsingle HRF library with shape `(num_hrfs, num_samples)`, sampled every 0.1 s from 0 to 50 s."""
    with files("prfmodel.data.hrf").joinpath("getcanonicalhrflibrary.tsv").open("r") as stream:
        library = np.loadtxt(stream, delimiter="\t", dtype=np.float64).T

    # Read-only, because the cached array is shared by every model instance
    library.setflags(write=False)

    return library


@doc
class HRFLibraryImpulse(BaseImpulse):
    """
    Impulse model that selects a hemodynamic response function (HRF) from a library of canonical HRFs.

    The library contains 20 canonical HRFs from GLMsingle [1]_ that span the range of HRF shapes observed in the
    Natural Scenes Dataset [2]_, ordered from early to late time-to-peak. The model has a single parameter:
    `hrf_index` selects the HRF from the library (from 0 to 19).

    Parameters
    ----------
    duration : float, default=50.0
        The duration of the impulse response (in seconds). The default covers the full library, whose slowest HRFs
        have not fully returned to zero after 32 seconds.
    %(impulse_offset)s
    %(impulse_resolution)s
    %(impulse_norm)s
    %(impulse_default_parameters)s

    Notes
    -----
    Parameter `hrf_index` is a discrete integer parameter that indexes the library. It must be a whole number
    between 0 and 19. It must not be optimized with stochastic gradient descent or least squares, use a grid search
    instead: Add `"hrf_index": np.arange(20)` to the parameter values of a
    :class:`~prfmodel.fitters.GridFitter` to find the best fitting HRF for each unit. Afterwards, keep the estimated
    `hrf_index` fixed in subsequent fitting steps.

    The library is sampled every 0.1 seconds from 0 to 50 seconds. The model resamples it at its time frames by
    linear interpolation. The response is zero at frames outside of this range.

    The library is distributed with GLMsingle under the BSD 3-Clause License, Copyright (c) 2021, Kendrick Kay. See
    :doc:`Copyright </copyright>`.

    References
    ----------
    .. [1] Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving
        the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife*, 11, e77599.
        https://doi.org/10.7554/eLife.77599
    .. [2] Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., ... & Kay, K. N.
        (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature
        Neuroscience*, 25(1), 116-126. https://doi.org/10.1038/s41593-021-00962-x

    Examples
    --------
    Predict the impulse responses of the first and last HRF in the library.

    >>> import pandas as pd
    >>> impulse_model = HRFLibraryImpulse()
    >>> params = pd.DataFrame({"hrf_index": [0, 19]})
    >>> resp = impulse_model(params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (2, 50)

    Find the best fitting HRF for each unit with a grid search.

    >>> import numpy as np
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.fitters import GridFitter
    >>> from prfmodel.models.prf import Gaussian2DPRFModel
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> model = Gaussian2DPRFModel(impulse_model=HRFLibraryImpulse(), scaling_model=None)
    >>> fitter = GridFitter(model=model, stimulus=stimulus)
    >>> data = np.random.default_rng(0).standard_normal((1, stimulus.design.shape[0]))
    >>> param_values = {"mu_y": [0.0], "mu_x": [0.0], "sigma": [1.0], "hrf_index": np.arange(20)}
    >>> history, params_grid = fitter.fit(data, param_values)
    >>> print(list(params_grid.columns))
    ['mu_y', 'mu_x', 'sigma', 'hrf_index']

    """

    def __init__(
        self,
        duration: float = 50.0,
        offset: float = 0.0,
        resolution: float | None = None,
        norm: str | None = "sum",
        default_parameters: dict[str, float] | None = None,
    ):
        super().__init__(duration, offset, resolution, norm, default_parameters)

        # Keyed by dtype and resolution rather than by dtype alone: 'at_resolution' makes shallow copies that
        # would otherwise share this cache and read a library resampled at another resolution.
        self._library: dict[tuple[str, float], np.ndarray] = {}

    @property
    def parameter_names(self) -> list[str]:
        """Parameter names are: `hrf_index`."""
        return ["hrf_index"]

    @doc
    def check_parameter_values(self, parameters: pd.DataFrame) -> None:
        """Check that the parameter values lie inside the domain the model is defined on.

        If not already present, adds default parameter columns to ``parameters``.

        Parameters
        ----------
        %(parameters)s

        Raises
        ------
        ValueError
            When `hrf_index` is not a whole number or not between 0 and 19.

        """
        parameters = self._join_default_parameters(parameters)

        if "hrf_index" in parameters.columns:
            hrf_index = parameters["hrf_index"].to_numpy()

            if not np.all(hrf_index == np.floor(hrf_index)):
                msg = "Parameter 'hrf_index' must be a whole number because it indexes the HRF library"
                raise ValueError(msg)

            if ((hrf_index < 0) | (hrf_index >= HRF_LIBRARY_SIZE)).any():
                msg = f"Parameter 'hrf_index' must be between 0 and {HRF_LIBRARY_SIZE - 1}"
                raise ValueError(msg)

        super().check_parameter_values(parameters)

    @doc
    def get_library(self, dtype: str | None = None) -> Tensor:
        """
        Get the HRF library resampled at the time frames of the model.

        Parameters
        ----------
        %(dtype)s

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The unnormalized HRF library with shape `(20, num_frames)`.

        """
        dtype = get_dtype(dtype)
        key = (dtype, self._sampling_resolution)

        if key not in self._library:
            library = _load_hrf_library()
            library_times = np.arange(library.shape[1]) * HRF_LIBRARY_RESOLUTION
            # Built in float64 rather than from 'get_frames', so that the interpolation does not depend on dtype
            times = np.arange(self.num_frames) * self._sampling_resolution + self.offset
            resampled = np.stack([np.interp(times, library_times, hrf, left=0.0, right=0.0) for hrf in library])
            self._library[key] = resampled.astype(dtype)

        return ops.convert_to_tensor(self._library[key], dtype=dtype)

    @doc
    def call(self, parameters: TensorFrame) -> Tensor:
        """
        Predict the impulse response.

        Parameters
        ----------
        %(parameters_tensors)s :attr:`default_parameters` must already be merged in.

        Returns
        -------
        :data:`prfmodel.typing.Tensor`
            The predicted impulse response with shape `(num_units, num_frames)`.

        """
        parameters = self._join_default_parameters(parameters)
        library = self.get_library(parameters.dtype)

        # Selected with 'ops.take' so that it also works when 'hrf_index' is a tensor inside a compiled function.
        # Gathering by index admits no gradient, so 'hrf_index' is estimable by grid search only.
        hrf_index = ops.cast(ops.round(parameters[["hrf_index"]]), "int32")
        response = ops.take(library, ops.reshape(hrf_index, (-1,)), axis=0)

        return normalize_response(response, self.norm)
