"""Divisive normalization population receptive field models."""

import pandas as pd
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.regressors.base import BaseRegressors
from prfmodel.scaling import Baseline
from prfmodel.scaling.base import BaseScaling
from ._gaussian import Gaussian2DPRFResponse
from ._stimulus_encoding import PRFStimulusEncoder
from .canonical import DivNormPRFModel


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
    %(model_encoding_prf)s
    %(model_impulse)s
    scaling_model : BaseScaling or type or None, default=Baseline, optional
        A scaling model class or instance. Model classes will be instantiated during initialization.
        The default creates a :class:`~prfmodel.scaling.Baseline` instance.
    %(model_regressors)s

    Notes
    -----
    With :math:`a = \text{amplitude\_activation}`, :math:`b = \text{baseline\_activation}`,
    :math:`c = \text{amplitude\_normalization}`, and :math:`d = \text{baseline\_normalization}`,
    the predicted response is:

    .. math::

        p_{\text{DN}} = \frac{(a G_1 \cdot S + b)}{(c G_2 \cdot S + d)} - \frac{b}{d}

    The :math:`-b/d` term ensures a zero response in the absence of a stimulus.

    Using the default impulse and scaling models, the following columns are expected in the
    :class:`pandas.DataFrame` passed as the ``parameters`` argument to :meth:`__call__`:

    .. list-table::
       :header-rows: 1
       :widths: 26 12 47

       * - Parameter
         - Model
         - Description
       * - ``mu_x``
         - pRF
         - Shared x-coordinate of both Gaussian centers.
       * - ``mu_y``
         - pRF
         - Shared y-coordinate of both Gaussian centers.
       * - ``sigma_activation``
         - pRF
         - Standard deviation of the activation Gaussian.
       * - ``sigma_normalization``
         - pRF
         - Standard deviation of the normalization Gaussian (must be >= ``sigma_activation``).
       * - ``delay``
         - Impulse
         - Peak time of the positive gamma component (in seconds; optional).
       * - ``dispersion``
         - Impulse
         - Rate parameter of the positive gamma component (optional).
       * - ``undershoot``
         - Impulse
         - Peak time of the negative gamma component (in seconds; optional).
       * - ``u_dispersion``
         - Impulse
         - Rate parameter of the negative gamma component (optional).
       * - ``ratio``
         - Impulse
         - Weight of the negative gamma component (optional).
       * - ``weight_deriv``
         - Impulse
         - Weight of the derivative component.
       * - ``amplitude_activation``
         - Scaling
         - Amplitude of the activation response (:math:`a`).
       * - ``baseline_activation``
         - Scaling
         - Baseline of the activation response (:math:`b`).
       * - ``amplitude_normalization``
         - Scaling
         - Amplitude of the normalization response (:math:`c`).
       * - ``baseline_normalization``
         - Scaling
         - Baseline of the normalization response (:math:`d`; must be > 0).

    """

    def __init__(
        self,
        encoding_model: BaseStimulusEncoder | type[BaseStimulusEncoder] = PRFStimulusEncoder,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        scaling_model: BaseScaling | type[BaseScaling] | None = Baseline,
        regressors_model: BaseRegressors | list[BaseRegressors] | None = None,
    ):
        super().__init__(
            activation_prf_model=Gaussian2DPRFResponse(),
            normalization_prf_model=Gaussian2DPRFResponse(),
            shared_params=["mu_x", "mu_y"],
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=scaling_model,
            regressors_model=regressors_model,
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
