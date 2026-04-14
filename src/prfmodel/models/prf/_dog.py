"""Difference of Gaussians population receptive field models."""

import pandas as pd
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseEncoder
from prfmodel.scaling import DoGAmplitude
from prfmodel.scaling.base import BaseTemporal
from ._gaussian import Gaussian2DPRFResponse
from ._stimulus_encoding import PRFStimulusEncoder
from .canonical import CenterSurroundPRFModel


class DoG2DPRFModel(CenterSurroundPRFModel):
    r"""
    Two-dimensional difference of Gaussians population receptive field model.

    Runs two Gaussian 2D PRF responses (center and surround) through stimulus encoding and impulse
    response convolution independently, then combines them as a linear model.

    Parameters
    ----------
    %(model_encoding)s
    %(model_impulse)s
    %(model_temporal)s

    Notes
    -----
    The simple composite model follows five steps [1]_:

    1. The center and surround 2D Gaussian population receptive field response models make separate predictions for
        the stimulus grid. The two response models have the same center but different sizes.
    2. The encoding model encodes each response with the stimulus design.
    3. A impulse response model generates an impulse response.
    4. Each encoded response is convolved with the impulse response.
    5. The temporal model modifies the convolved response. By default it subtracts the surround from the center
        response after multiplying the responses with separate amplitude parameters.

    Let :math:`p_{\text{center}}(t)` and :math:`p_{\text{surround}}(t)` be the predicted temporal
    responses for the center and surround Gaussians. With :math:`a_c = \text{amplitude\_center}`,
    :math:`a_s = \text{amplitude\_surround}`, and :math:`\beta = \text{baseline}`, the predicted
    response is:

    .. math::

        y(t) = a_c \, p_{\text{center}}(t) + a_s \, p_{\text{surround}}(t) + \beta

    References
    ----------
    .. [1] Zuiderbaan, W., Harvey, B. M., & Dumoulin, S. O. (2012). Modeling center-surround configurations in
        population receptive fields using fMRI. *Journal of Vision*, 12(3), 10. https://doi.org/10.1167/12.3.10

    Examples
    --------
    Predict a model response for multiple units.

    >>> import pandas as pd
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> from prfmodel.models import DoG2DPRFModel
    >>> stimulus = load_2d_prf_bar_stimulus()
    >>> print(stimulus)
    PRFStimulus(design=array[200, 101, 101], grid=array[101, 101, 2], dimension_labels=['y', 'x'])
    >>> model = DoG2DPRFModel()
    >>> # Define all model parameters for 3 units
    >>> params = pd.DataFrame({
    ...     # DoG parameters
    ...     "mu_x": [0.0, 1.0, 0.0],
    ...     "mu_y": [1.0, 0.0, 0.0],
    ...     "sigma_center": [1.0, 1.5, 2.0],
    ...     "sigma_surround": [5.0, 7.5, 10.0],
    ...     # Impulse model parameters
    ...     "delay": [6.0, 6.0, 6.0],
    ...     "dispersion": [0.9, 0.9, 0.9],
    ...     "undershoot": [12.0, 12.0, 12.0],
    ...     "u_dispersion": [0.9, 0.9, 0.9],
    ...     "ratio": [0.48, 0.48, 0.48],
    ...     "weight_deriv": [0.5, 0.5, 0.5],
    ...     # Temporal model parameters
    ...     "amplitude_center": [2.0, 1.2, 0.1],
    ...     "amplitude_surround": [-0.5, -0.3, -0.1],
    ...     "baseline": [0.1, -0.1, 0.5],
    ... })
    >>> # Predict model response
    >>> resp = model(stimulus, params)
    >>> print(resp.shape)  # (num_units, num_frames)
    (3, 200)

    """

    def __init__(
        self,
        encoding_model: BaseEncoder | type[BaseEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DoGAmplitude,
    ):
        super().__init__(
            prf_model=Gaussian2DPRFResponse(),
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
            change_params=["sigma"],
        )


def init_dog_from_gaussian(
    gaussian_params: pd.DataFrame,
    sigma_ratio: float = 5.0,
    sigma_surround: float | None = None,
) -> pd.DataFrame:
    """
    Initialize DoG model parameters from fitted Gaussian model parameters.

    Converts the output of a fitted :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`
    into starting parameters for a :class:`~prfmodel.models.dog.DoG2DPRFModel`, suitable for
    subsequent SGD.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`.
        Must contain columns: ``sigma`` and ``amplitude`` (plus all shared columns).
    sigma_ratio : float, default=5.0
        Ratio used to set the surround size: ``sigma_surround = sigma_center * sigma_ratio``.
        Ignored when ``sigma_surround`` is provided.
    sigma_surround : float, optional
        Fixed surround size applied to all rows. Must be >= ``sigma`` for every row in
        ``gaussian_params``. When provided, overrides ``sigma_ratio``.

    Returns
    -------
    pandas.DataFrame
        DataFrame of DoG initial parameters with columns:
        ``sigma_center`` (= ``sigma``), ``sigma_surround``,
        ``amplitude_center`` (= ``amplitude``), ``amplitude_surround`` (= 0.0),
        plus all shared columns unchanged.

    Raises
    ------
    ValueError
        If ``sigma_surround`` is smaller than ``sigma`` for any row in ``gaussian_params``.

    Examples
    --------
    >>> import pandas as pd
    >>> from prfmodel.models import init_dog_from_gaussian
    >>> gaussian_params = pd.DataFrame({
    ...     "mu_x": [0.0, 1.0],
    ...     "mu_y": [0.0, -1.0],
    ...     "sigma": [1.0, 2.0],
    ...     "amplitude": [1.0, -1.0],
    ...     "baseline": [0.0, 0.1],
    ... })
    >>> dog_params = init_dog_from_gaussian(gaussian_params, sigma_ratio=3.0)
    >>> print(dog_params["sigma_center"].tolist())
    [1.0, 2.0]
    >>> print(dog_params["sigma_surround"].tolist())
    [3.0, 6.0]

    Notes
    -----
    ``amplitude_surround`` is initialized to ``0.0``, which is the boundary of the constraint
    ``amplitude_surround < 0`` enforced by a :class:`~prfmodel.adapter.ParameterConstraint`
    with ``upper=0.0``. The constraint transform maps ``amplitude_surround=0`` to optimizer
    variable ``raw=-1.0`` (no NaN), so SGD starts cleanly near zero and moves negative.

    """
    dog_params = gaussian_params.copy()
    dog_params["sigma_center"] = dog_params["sigma"]

    if sigma_surround is not None:
        if (gaussian_params["sigma"] > sigma_surround).any():
            min_sigma_center = gaussian_params["sigma"].max()
            msg = (
                f"sigma_surround ({sigma_surround}) must be >= sigma_center for all rows, "
                f"but max sigma_center is {min_sigma_center}"
            )
            raise ValueError(msg)
        dog_params["sigma_surround"] = sigma_surround
    else:
        dog_params["sigma_surround"] = dog_params["sigma"] * sigma_ratio

    dog_params["amplitude_center"] = dog_params["amplitude"]
    dog_params["amplitude_surround"] = 0.0
    return dog_params.drop(columns=["sigma", "amplitude"])
