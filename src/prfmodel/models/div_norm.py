"""Divisive normalization population receptive field models."""

from typing import cast
import pandas as pd
from keras import ops
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.impulse.base import BaseImpulse
from prfmodel.scaling import DivNormAmplitude
from prfmodel.scaling.base import BaseTemporal
from prfmodel.stimuli.prf import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import get_dtype
from .base import BaseComposite
from .base import BaseEncoder
from .base import BaseResponse
from .encoding import PRFStimulusEncoder
from .gaussian import Gaussian2DPRFResponse


class DivNormPRFModel(BaseComposite[PRFStimulus]):
    r"""
    Divisive normalization population receptive field model.

    Receives two independent pRF responses (activation and normalization) and combines them
    via the divisive normalization formula.
    Parameters that should be shared between both responses (e.g. the pRF centre
    ``mu_x``, ``mu_y``) are listed in ``shared_params`` and appear once in
    :attr:`parameter_names` without a suffix. All remaining pRF parameters are suffixed
    with ``_activation`` or ``_normalization`` respectively.

    Parameters
    ----------
    activation_prf_model : BaseResponse
        pRF response model (activation).
    normalization_prf_model : BaseResponse
        pRF response model (normalization).
    shared_params : list of str, default=["mu_x", "mu_y"]
        Names of pRF parameters that are shared between the two responses.  Each name
        must appear in *both* ``activation_prf_model.parameter_names`` and
        ``normalization_prf_model.parameter_names``.
    encoding_model : BaseEncoder or type, default=PRFStimulusEncoder
        An encoding model class or instance.
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DivNormAmplitude
        A temporal model class or instance.

    Notes
    -----
    The predicted response is:

    .. math::

        p_{\text{DN}} = \frac{(a R_1 \cdot S + b)}{(c R_2 \cdot S + d)} - \frac{b}{d}

    Where `R_1` and `R_2` are the activation and normalization pRF responses, `S` is the stimulus.
    The :math:`-b/d` term ensures a zero response in the absence of a stimulus.

    """

    def __init__(  # noqa: PLR0913
        self,
        activation_prf_model: BaseResponse,
        normalization_prf_model: BaseResponse,
        shared_params: list[str] | None = None,
        encoding_model: BaseEncoder | type[BaseEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DivNormAmplitude,
    ):
        if shared_params is None:
            shared_params = ["mu_x", "mu_y"]

        if encoding_model is not None and isinstance(encoding_model, type):
            encoding_model = encoding_model()

        if impulse_model is not None and isinstance(impulse_model, type):
            impulse_model = impulse_model()

        if temporal_model is not None and isinstance(temporal_model, type):
            temporal_model = temporal_model()

        act_names = activation_prf_model.parameter_names
        norm_names = normalization_prf_model.parameter_names
        invalid = [p for p in shared_params if p not in act_names or p not in norm_names]
        if invalid:
            msg = (
                f"DivNormPRFModel: shared_params {invalid} not found in both "
                f"activation_prf_model.parameter_names {act_names} and "
                f"normalization_prf_model.parameter_names {norm_names}"
            )
            raise ValueError(msg)

        self._shared_params = shared_params

        super().__init__(
            activation_prf_model=activation_prf_model,
            normalization_prf_model=normalization_prf_model,
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

    @property
    def parameter_names(self) -> list[str]:
        """
        Names of parameters used by the model.

        Shared parameters appear once (no suffix). Response-specific parameters are suffixed
        with ``_activation`` or ``_normalization``.

        """
        shared = set(self._shared_params)
        act_model = cast("BaseResponse", self.models["activation_prf_model"])
        norm_model = cast("BaseResponse", self.models["normalization_prf_model"])

        param_names: list[str] = []

        # Activation model params: shared appear as-is, non-shared get _activation suffix
        for p in act_model.parameter_names:
            if p in shared:
                param_names.append(p)
            else:
                param_names.append(f"{p}_activation")

        # Normalization model non-shared params get _normalization suffix
        param_names.extend(f"{p}_normalization" for p in norm_model.parameter_names if p not in shared)

        # Encoding, impulse, and temporal model params
        for key, model in self.models.items():
            if key not in ("activation_prf_model", "normalization_prf_model") and model is not None:
                param_names.extend(model.parameter_names)

        return list(dict.fromkeys(param_names))

    def _predict_single_response(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        suffix: str,
        dtype: str,
    ) -> Tensor:
        """Run one pRF response (pRF response -> encoding -> optional impulse convolution)."""
        prf_model = cast("BaseResponse", self.models[f"{suffix}_prf_model"])
        shared = set(self._shared_params)

        # Build a parameter slice for this pRF model: copy all params, then
        # overwrite non-shared params from the suffixed columns.
        params_single = parameters.copy()
        for p in prf_model.parameter_names:
            if p not in shared:
                params_single[p] = parameters[f"{p}_{suffix}"]

        response = prf_model(stimulus, params_single, dtype=dtype)
        encoding_model = cast("BaseEncoder", self.models["encoding_model"])
        response = encoding_model(stimulus, response, parameters, dtype=dtype)

        if self.models["impulse_model"] is not None:
            impulse_model = cast("BaseImpulse", self.models["impulse_model"])
            impulse_response = impulse_model(parameters, dtype=dtype)
            response = convolve_prf_impulse_response(response, impulse_response, dtype=dtype)

        return response

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
            is the activation response and index 1 is the normalization response.

        """
        dtype = get_dtype(dtype)
        p1 = self._predict_single_response(stimulus, parameters, "activation", dtype)
        p2 = self._predict_single_response(stimulus, parameters, "normalization", dtype)
        return ops.stack([p1, p2], axis=1)

    def __call__(
        self,
        stimulus: PRFStimulus,
        parameters: pd.DataFrame,
        dtype: str | None = None,
    ) -> Tensor:
        """
        Predict the DN model response to a stimulus.

        Parameters
        ----------
        stimulus : PRFStimulus
            Population receptive field stimulus object.
        parameters : pandas.DataFrame
            Dataframe with model parameters; one row per voxel.
        dtype : str, optional
            The dtype of the prediction result. If ``None`` (the default), uses the dtype from
            :func:`prfmodel.utils.get_dtype`.

        Returns
        -------
        Tensor
            Model predictions of shape (num_voxels, num_frames).

        """
        dtype = get_dtype(dtype)
        stacked = self.predict_responses(stimulus, parameters, dtype=dtype)

        if self.models["temporal_model"] is not None:
            temporal_model = cast("BaseTemporal", self.models["temporal_model"])
            return temporal_model(stacked, parameters, dtype=dtype)

        # TODO: Is this a sensible default? Or what would word best in the absence of a temporal model?
        return stacked[:, 0] / stacked[:, 1]


class DivNormGaussian2DPRFModel(DivNormPRFModel):
    r"""
    Divisive normalization pRF model with isotropic 2D Gaussian responses.

    Convenience subclass of :class:`DivNormPRFModel` that uses
    :class:`~prfmodel.models.gaussian.Gaussian2DPRFResponse` as the pRF model. Both the
    activation and normalization pRFs are isotropic 2D Gaussians centered on the same
    position (``mu_x``, ``mu_y``) but with different sizes (``sigma_activation``,
    ``sigma_normalization``) and different amplitudes (``amplitude_activation``,
    ``amplitude_normalization``).

    Parameters
    ----------
    encoding_model : BaseEncoder or type, default=PRFStimulusEncoder
        An encoding model class or instance.
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DivNormAmplitude
        A temporal model class or instance.

    Notes
    -----
    With :math:`a = \text{amplitude\_activation}`, :math:`b = \text{baseline\_activation}`,
    :math:`c = \text{amplitude\_normalization}`, and :math:`d = \text{baseline\_normalization}`,
    the predicted response is:

    .. math::

        p_{\text{DN}} = \frac{(a G_1 \cdot S + b)}{(c G_2 \cdot S + d)} - \frac{b}{d}

    The :math:`-b/d` term ensures a zero response in the absence of a stimulus.

    """

    def __init__(
        self,
        encoding_model: BaseEncoder | type[BaseEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DivNormAmplitude,
    ):
        super().__init__(
            activation_prf_model=Gaussian2DPRFResponse(),
            normalization_prf_model=Gaussian2DPRFResponse(),
            shared_params=["mu_x", "mu_y"],
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )


def init_dn_from_gaussian(  # noqa: PLR0913
    gaussian_params: pd.DataFrame,
    sigma_ratio: float = 2.0,
    sigma_normalization: float | None = None,
    baseline_activation: float | None = None,
    amplitude_normalization: float = 1.0,
    baseline_normalization: float = 1.0,
) -> pd.DataFrame:
    """
    Initialize DN model parameters from fitted Gaussian model parameters.

    Converts the output of a fitted :class:`~prfmodel.models.gaussian.Gaussian2DPRFModel`
    into starting parameters for a :class:`DivNormGaussian2DPRFModel`, suitable for
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
    baseline_activation : float, optional
        Initial value for ``baseline_activation`` (b in the DN formula). If ``None`` (the
        default), the value is taken from the ``baseline`` column of ``gaussian_params``; a
        ``ValueError`` is raised if that column is absent. When an explicit float is given it
        is always used, regardless of whether a ``baseline`` column is present.
    amplitude_normalization : float, default=1.0
        Initial value for ``amplitude_normalization`` (c in the DN formula). Controls the
        scaling of the normalization Gaussian.
    baseline_normalization : float, default=1.0
        Initial value for ``baseline_normalization`` (d in the DN formula). Controls the
        normalization baseline in the denominator. Must be > 0 to avoid division by zero.

    Returns
    -------
    pandas.DataFrame
        DataFrame of DN initial parameters with columns:
        ``sigma_activation`` (= ``sigma``), ``sigma_normalization``,
        ``amplitude_activation`` (= ``amplitude``), ``baseline_activation`` (b, = ``baseline``
        if present), ``amplitude_normalization`` (c), ``baseline_normalization`` (d),
        plus all shared columns unchanged. The ``sigma``, ``amplitude``, and ``baseline``
        (if present) columns are dropped.

    Raises
    ------
    ValueError
        If ``sigma_ratio`` is less than 1.0.
    ValueError
        If ``sigma_normalization`` is smaller than ``sigma`` for any row in ``gaussian_params``.
    ValueError
        If ``baseline_activation`` is ``None`` and ``gaussian_params`` does not contain a
        ``baseline`` column.

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
    if baseline_activation is None:
        if "baseline" not in dn_params.columns:
            msg = (
                "`baseline_activation` is None and gaussian_params does not contain a 'baseline' column. "
                "Provide an explicit `baseline_activation` value."
            )
            raise ValueError(msg)
        dn_params["baseline_activation"] = dn_params["baseline"]
    else:
        dn_params["baseline_activation"] = baseline_activation
    dn_params["amplitude_normalization"] = amplitude_normalization
    dn_params["baseline_normalization"] = baseline_normalization

    cols_to_drop = ["sigma", "amplitude"] + (["baseline"] if "baseline" in dn_params.columns else [])
    return dn_params.drop(columns=cols_to_drop)


def init_dn_from_dog(
    dog_params: pd.DataFrame,
    baseline_activation: float | None = None,
    baseline_normalization: float = 1.0,
) -> pd.DataFrame:
    """
    Initialize DN model parameters from fitted DoG model parameters.

    Converts the output of a fitted :class:`~prfmodel.models.difference_of_gaussians.DoG2DPRFModel`
    into starting parameters for a :class:`DivNormGaussian2DPRFModel`, suitable for subsequent SGD.

    Parameters
    ----------
    dog_params : pandas.DataFrame
        DataFrame of fitted parameters from a ``DoG2DPRFModel``.
        Must contain columns: ``sigma_center``, ``sigma_surround``, ``amplitude_center``,
        and ``amplitude_surround`` (plus all shared columns).
    baseline_activation : float, optional
        Initial value for ``baseline_activation`` (b in the DN formula). If ``None`` (the
        default), the value is taken from the ``baseline`` column of ``dog_params``; a
        ``ValueError`` is raised if that column is absent. When an explicit float is given it
        is always used, regardless of whether a ``baseline`` column is present.
    baseline_normalization : float, default=1.0
        Initial value for ``baseline_normalization`` (d in the DN formula). Controls the
        normalization baseline in the denominator. Must be > 0 to avoid division by zero.

    Returns
    -------
    pandas.DataFrame
        DataFrame of DN initial parameters with columns:
        ``sigma_activation`` (= ``sigma_center``), ``sigma_normalization`` (= ``sigma_surround``),
        ``amplitude_activation`` (= ``amplitude_center``),
        ``amplitude_normalization`` (= ``abs(amplitude_surround)``; negated to start in the
        normalization regime — can go negative during SGD if unconstrained),
        ``baseline_activation`` (b, = ``baseline`` if present), ``baseline_normalization`` (d),
        plus all shared columns unchanged. The ``sigma_center``, ``sigma_surround``,
        ``amplitude_center``, ``amplitude_surround``, and ``baseline`` (if present) columns
        are dropped.

    Raises
    ------
    ValueError
        If ``baseline_activation`` is ``None`` and ``dog_params`` does not contain a
        ``baseline`` column.

    """
    dn_params = dog_params.copy()
    dn_params["sigma_activation"] = dn_params["sigma_center"]
    dn_params["sigma_normalization"] = dn_params["sigma_surround"]
    dn_params["amplitude_activation"] = dn_params["amplitude_center"]
    # NOTE: I am not sure about this, but in the DN paper I never saw negative `amplitude_normalization`,
    # and in the DoG model the surround amplitude is often negative, so I take the absolute value here ?????
    dn_params["amplitude_normalization"] = dn_params["amplitude_surround"].abs()

    if baseline_activation is None:
        if "baseline" not in dn_params.columns:
            msg = (
                "`baseline_activation` is None and dog_params does not contain a 'baseline' column. "
                "Provide an explicit `baseline_activation` value."
            )
            raise ValueError(msg)
        dn_params["baseline_activation"] = dn_params["baseline"]
    else:
        dn_params["baseline_activation"] = baseline_activation

    dn_params["baseline_normalization"] = baseline_normalization

    cols_to_drop = ["sigma_center", "sigma_surround", "amplitude_center", "amplitude_surround"]
    if "baseline" in dn_params.columns:
        cols_to_drop.append("baseline")
    return dn_params.drop(columns=cols_to_drop)
