"""Difference of Gaussians population receptive field models."""

import pandas as pd
from .base import BaseEncoder
from .base import BaseImpulse
from .base import BaseTemporal
from .composite import CenterSurroundPRFModel
from .encoding import PRFStimulusEncoder
from .gaussian import Gaussian2DPRFResponse
from .impulse import DerivativeTwoGammaImpulse
from .temporal import DoGAmplitude


class DoG2DPRFModel(CenterSurroundPRFModel):
    """
    Two-dimensional difference of Gaussians population receptive field model.

    Runs two Gaussian 2D PRF responses (center and surround) through stimulus encoding and impulse response convolution
    independently, then combines them as a linear model:

    y(t) = p1(t) * amplitude_center + p2(t) * amplitude_sorround + baseline

    Parameters
    ----------
     encoding_model : BaseEncoder or type, default=PRFStimulusEncoder
        An encoding model class or instance. Model classes will be instantiated during initialization. The
        default creates a :class:`~prfmodel.models.encoding.PRFStimulusEncoder` instance.
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DoGAmplitude
        A temporal model class or instance.

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
    sigma_sorround: float | None = None,
) -> pd.DataFrame:
    """
    Initialize DoG model parameters from fitted Gaussian model parameters.

    Converts the output of a fitted :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`
    into starting parameters for a :class:`DoG2DPRFModel`, suitable for subsequent SGD.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a ``Gaussian2DPRFModel``.
        Must contain columns: ``sigma`` and ``amplitude`` (plus all shared columns).
    sigma_ratio : float, default=5.0
        Ratio used to set the surround size: ``sigma_sorround = sigma_center * sigma_ratio``.
        Ignored when ``sigma_sorround`` is provided.
    sigma_sorround : float, optional
        Fixed surround size applied to all rows. Must be >= ``sigma`` for every row in
        ``gaussian_params``. When provided, overrides ``sigma_ratio``.

    Returns
    -------
    pandas.DataFrame
        DataFrame of DoG initial parameters with columns:
        ``sigma_center`` (= ``sigma``), ``sigma_sorround``,
        ``amplitude_center`` (= ``amplitude``), ``amplitude_sorround`` (= 0.0),
        plus all shared columns unchanged.

    Raises
    ------
    ValueError
        If ``sigma_sorround`` is smaller than ``sigma`` for any row in ``gaussian_params``.

    Notes
    -----
    ``amplitude_sorround`` is initialized to ``0.0``, which is the boundary of the constraint
    ``amplitude_sorround < 0`` enforced by a :class:`~prfmodel.adapter.ParameterConstraint`
    with ``upper=0.0``. The constraint transform maps ``amplitude_sorround=0`` to optimizer
    variable ``raw=-1.0`` (no NaN), so SGD starts cleanly near zero and moves negative.

    """
    dog_params = gaussian_params.copy()
    dog_params["sigma_center"] = dog_params["sigma"]

    if sigma_sorround is not None:
        if (gaussian_params["sigma"] > sigma_sorround).any():
            min_sigma_center = gaussian_params["sigma"].max()
            msg = (
                f"sigma_sorround ({sigma_sorround}) must be >= sigma_center for all rows, "
                f"but max sigma_center is {min_sigma_center}"
            )
            raise ValueError(msg)
        dog_params["sigma_sorround"] = sigma_sorround
    else:
        dog_params["sigma_sorround"] = dog_params["sigma"] * sigma_ratio

    dog_params["amplitude_center"] = dog_params["amplitude"]
    dog_params["amplitude_sorround"] = 0.0
    return dog_params.drop(columns=["sigma", "amplitude"])
