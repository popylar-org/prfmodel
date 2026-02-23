"""Difference of Gaussians population receptive field models."""

from .base import BaseImpulse
from .base import BaseTemporal
from .composite import DoGPRFModel
from .gaussian import Gaussian2DPRFResponse
from .impulse import DerivativeTwoGammaImpulse
from .temporal import DoGAmplitude


class DoG2DPRFModel(DoGPRFModel):
    """
    Two-dimensional difference of Gaussians population receptive field model.

    Runs two Gaussian pipelines (sigma1 and sigma2) through encode and convolve
    independently, then combines them as a linear model:
    y(t) = p1(t) * amplitude_1 + p2(t) * amplitude_2 + baseline

    Parameters
    ----------
    impulse_model : BaseImpulse or type or None, default=DerivativeTwoGammaImpulse
        An impulse response model class or instance.
    temporal_model : BaseTemporal or type or None, default=DoGAmplitude
        A temporal model class or instance.

    """

    def __init__(
        self,
        impulse_model: BaseImpulse | type[BaseImpulse] | None = DerivativeTwoGammaImpulse,
        temporal_model: BaseTemporal | type[BaseTemporal] | None = DoGAmplitude,
    ):
        super().__init__(
            prf_model=Gaussian2DPRFResponse(),
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )
