"""Divisive normalization population receptive field models."""

from typing import cast
import pandas as pd
from keras import ops
from prfmodel.stimuli.prf import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .base import BaseImpulse
from .base import BasePRFResponse
from .base import BaseTemporal
from .composite import CenterSurroundPRFModel
from .gaussian import Gaussian2DPRFResponse
from .impulse import DerivativeTwoGammaImpulse
from .temporal import DivNormAmplitude


class DivNormPRFModel(CenterSurroundPRFModel):
    """
    Two-dimensional divisive normalization population receptive field model.

    Both the activation and normalization pRFs are isotropic 2D Gaussians centered on the
    same position (``mu_x``, ``mu_y``) but with different sizes (``sigma_activation``,
    ``sigma_normalization``) and different amplitudes (``amplitude_activation``,
    ``amplitude_normalization``). The predicted neural response is:

    p_DN(t) = (a * G1·S + b) / (c * G2·S + d) - b/d

    where G1 and G2 are the activation and normalization Gaussians, ``a`` and ``b`` are the
    activation amplitude and constant, and ``c`` and ``d`` are the normalization amplitude
    and constant. ``b`` and ``d`` together modulate suppression and compression in the model.
    G*·S denotes the element-wise product of the Gaussian and the stimulus, summed over all
    spatial coordinates.

    Parameters
    ----------
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DivNormAmplitude
        A temporal model class or instance.

    """

    def __init__(
        self,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DivNormAmplitude,
    ):
        super().__init__(
            prf_model=Gaussian2DPRFResponse(),
            impulse_model=impulse_model,
            temporal_model=temporal_model,
            change_params=["sigma"],
        )

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Overrides the parent to use ``activation``/``normalization`` suffixes.

        """
        prf_model = cast("BasePRFResponse", self.models["prf_model"])
        prf_params = prf_model.parameter_names.copy()

        for param in self._change_params:
            idx = prf_params.index(param)
            prf_params[idx : idx + 1] = [f"{param}_activation", f"{param}_normalization"]

        param_names = prf_params

        for key, model in self.models.items():
            if key != "prf_model" and model is not None:
                param_names.extend(model.parameter_names)

        return list(dict.fromkeys(param_names))

    def predict_responses(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict each of the two responses before applying the DN formula.

        Returns
        -------
        Tensor
            Stacked predictions of shape (num_voxels, 2, num_frames), where axis 1 index 0
            is the activation response (G1·S) and index 1 is the normalization response (G2·S).

        """
        dtype = get_dtype(dtype)
        p1 = self._predict_single_response(stimulus, parameters, "activation", dtype)
        p2 = self._predict_single_response(stimulus, parameters, "normalization", dtype)
        return ops.stack([p1, p2], axis=1)


def init_dn_from_gaussian(  # noqa: PLR0913
    gaussian_params: pd.DataFrame,
    sigma_ratio: float = 2.0,
    sigma_normalization: float | None = None,
    activation_constant: float = 0.0,
    amplitude_normalization: float = 1.0,
    normalization_constant: float = 1.0,
) -> pd.DataFrame:
    """
    Initialize DN model parameters from fitted Gaussian model parameters.

    Converts the output of a fitted :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`
    into starting parameters for a :class:`DivNormPRFModel`, suitable for
    subsequent SGD.

    Parameters
    ----------
    gaussian_params : pandas.DataFrame
        DataFrame of fitted parameters from a ``Gaussian2DPRFModel``.
        Must contain columns: ``sigma`` and ``amplitude`` (plus all shared columns).
    sigma_ratio : float, default=2.0
        Ratio used to set the normalization size (phi = sigma_normalization / sigma_activation in the paper):
        ``sigma_normalization = sigma_activation * sigma_ratio``.
        Must be >= 1.0, as the normalization pRF must be at least as large as the activation
        pRF. Ignored when ``sigma_normalization`` is provided.
    sigma_normalization : float, optional
        Fixed normalization size applied to all rows. Must be >= ``sigma`` for every row in
        ``gaussian_params``. When provided, overrides ``sigma_ratio``.
    activation_constant : float, default=0.0
        Initial value for ``activation_constant`` (b in the DN formula). Controls the
        activation baseline in the numerator.
    amplitude_normalization : float, default=1.0
        Initial value for ``amplitude_normalization`` (c in the DN formula). Controls the
        scaling of the normalization Gaussian.
    normalization_constant : float, default=1.0
        Initial value for ``normalization_constant`` (d in the DN formula). Controls the
        normalization baseline in the denominator. Must be > 0 to avoid division by zero.

    Returns
    -------
    pandas.DataFrame
        DataFrame of DN initial parameters with columns:
        ``sigma_activation`` (= ``sigma``), ``sigma_normalization``,
        ``amplitude_activation`` (= ``amplitude``), ``activation_constant`` (b),
        ``amplitude_normalization`` (c), ``normalization_constant`` (d),
        plus all shared columns unchanged.

    Raises
    ------
    ValueError
        If ``sigma_ratio`` is less than 1.0.
    ValueError
        If ``sigma_normalization`` is smaller than ``sigma`` for any row in ``gaussian_params``.

    """
    if sigma_ratio < 1.0:
        msg = (
            f"sigma_ratio must be >= 1.0 (got {sigma_ratio}), as the normalization pRF "
            "must be at least as large as the activation pRF."
        )
        raise ValueError(msg)

    dn_params = gaussian_params.copy()
    dn_params["sigma_activation"] = dn_params["sigma"]

    if sigma_normalization is not None:
        if (gaussian_params["sigma"] > sigma_normalization).any():
            max_sigma_activation = gaussian_params["sigma"].max()
            msg = (
                f"sigma_normalization ({sigma_normalization}) must be >= sigma_activation for all rows, "
                f"but max sigma_activation is {max_sigma_activation}"
            )
            raise ValueError(msg)
        dn_params["sigma_normalization"] = sigma_normalization
    else:
        dn_params["sigma_normalization"] = dn_params["sigma"] * sigma_ratio

    dn_params["amplitude_activation"] = dn_params["amplitude"]
    dn_params["activation_constant"] = activation_constant
    dn_params["amplitude_normalization"] = amplitude_normalization
    dn_params["normalization_constant"] = normalization_constant

    return dn_params.drop(columns=["sigma", "amplitude"])
