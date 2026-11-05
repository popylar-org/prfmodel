"""Tests that canonical models sample their impulse submodels at the frame duration of the stimulus."""

import warnings
from collections.abc import Callable
import numpy as np
import pandas as pd
import pytest
from prfmodel.exceptions import ResolutionMismatchWarning
from prfmodel.fitters import GridFitter
from prfmodel.fitters import LeastSquaresFitter
from prfmodel.fitters import SGDFitter
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import SustainedImpulse
from prfmodel.impulse import TransientImpulse
from prfmodel.models.base import BaseCanonical
from prfmodel.models.csf import CSFModel
from prfmodel.models.prf import DelayedNormGaussian2DPRFModel
from prfmodel.models.prf import DivNormGaussian2DPRFModel
from prfmodel.models.prf import Gaussian2DCSTPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import CSFStimulus
from prfmodel.stimuli import PRFStimulus
from prfmodel.stimuli import Stimulus

FRAME_DURATION = 1.5

HRF_PARAMETERS = {
    "delay": [6.0, 7.0],
    "dispersion": [0.9, 1.0],
    "undershoot": [12.0, 11.0],
    "u_dispersion": [0.9, 1.0],
    "ratio": [0.48, 0.48],
    "weight_deriv": [-0.5, 0.0],
}

PRF_PARAMETERS = {"mu_x": [0.0, 1.0], "mu_y": [1.0, 0.0], "sigma": [1.0, 1.5]}

ModelFactory = Callable[[float | None], BaseCanonical]


def _gaussian(resolution: float | None) -> BaseCanonical:
    return Gaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=resolution))


def _div_norm(resolution: float | None) -> BaseCanonical:
    return DivNormGaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=resolution))


def _delayed_norm(resolution: float | None) -> BaseCanonical:
    return DelayedNormGaussian2DPRFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=resolution))


def _cst(resolution: float | None) -> BaseCanonical:
    return Gaussian2DCSTPRFModel(
        sustained_model=SustainedImpulse(resolution=resolution),
        transient_model=TransientImpulse(resolution=resolution),
        impulse_model=DerivativeTwoGammaImpulse(resolution=resolution),
    )


def _csf(resolution: float | None) -> BaseCanonical:
    return CSFModel(impulse_model=DerivativeTwoGammaImpulse(resolution=resolution))


def _prf_stimulus(frame_duration: float) -> PRFStimulus:
    return PRFStimulus.create_2d_bar_stimulus(
        num_frames=40,
        width=16,
        height=16,
        pixel_size=0.5,
        frame_duration=frame_duration,
    )


def _csf_stimulus(frame_duration: float) -> CSFStimulus:
    return CSFStimulus(
        sf=np.array([0.5, 1.0, 3.0, 6.0, 12.0, 3.0, 1.0, 0.5] * 3),
        contrast=np.array([0.05, 0.1, 0.2, 0.4, 0.8, 0.4, 0.1, 0.05] * 3),
        frame_duration=frame_duration,
    )


MODEL_CASES = [
    pytest.param(
        _gaussian,
        _prf_stimulus,
        {**PRF_PARAMETERS, **HRF_PARAMETERS, "baseline": [0.0, 0.5], "amplitude": [1.0, 2.0]},
        id="gaussian",
    ),
    pytest.param(
        _div_norm,
        _prf_stimulus,
        {
            "mu_x": [0.0, 1.0],
            "mu_y": [1.0, 0.0],
            "sigma_activation": [1.0, 1.5],
            "sigma_normalization": [2.0, 3.0],
            **HRF_PARAMETERS,
            "amplitude_activation": [1.1, 1.0],
            "baseline_activation": [0.0, 0.1],
            "amplitude_normalization": [10.0, 5.0],
            "baseline_normalization": [20.0, 10.0],
            "baseline": [-0.5, 0.5],
        },
        id="div_norm",
    ),
    pytest.param(
        _delayed_norm,
        _prf_stimulus,
        {
            **PRF_PARAMETERS,
            **HRF_PARAMETERS,
            "n": [2.0, 2.0],
            "dispersion_normalization": [2.0, 3.0],
            "sigma_saturation": [1.0, 1.0],
            "amplitude": [1.0, 1.0],
            "baseline": [0.0, 0.0],
        },
        id="delayed_norm",
    ),
    pytest.param(
        _cst,
        _prf_stimulus,
        {
            **PRF_PARAMETERS,
            **HRF_PARAMETERS,
            "n": [0.5, 0.8],
            "amplitude_sustained": [1.0, 0.5],
            "amplitude_transient": [0.5, 1.0],
            "time_to_peak": [4.0, 5.0],
            "baseline": [-0.5, 0.5],
        },
        id="cst",
    ),
    pytest.param(
        _csf,
        _csf_stimulus,
        {
            "cs_peak": [100.0, 150.0],
            "sf_peak": [2.5, 3.0],
            "width_r": [1.28, 1.5],
            "slope_crf": [2.0, 3.0],
            **HRF_PARAMETERS,
            "baseline": [0.0, 0.5],
            "amplitude": [1.0, 2.0],
        },
        id="csf",
    ),
]


@pytest.mark.parametrize(("make_model", "make_stimulus", "parameters"), MODEL_CASES)
def test_default_impulse_follows_frame_duration(
    make_model: ModelFactory,
    make_stimulus: Callable[[float], Stimulus],
    parameters: dict[str, list[float]],
):
    """Test that impulse submodels without a resolution are sampled at the frame duration of the stimulus.

    The prediction must equal that of a model whose impulse submodels were given the frame duration explicitly,
    which also covers time axes that are derived from an impulse submodel, like the normalization kernel of the
    delayed normalization model, and the temporal channels of the CST model.

    """
    stimulus = make_stimulus(FRAME_DURATION)
    params = pd.DataFrame(parameters)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionMismatchWarning)

        response_default = make_model(None)(stimulus, params)
        response_explicit = make_model(FRAME_DURATION)(stimulus, params)

    np.testing.assert_allclose(response_default, response_explicit)

    with pytest.warns(ResolutionMismatchWarning):
        response_mismatch = make_model(1.0)(stimulus, params)

    assert not np.allclose(response_default, response_mismatch)


@pytest.mark.parametrize(("make_model", "make_stimulus", "parameters"), MODEL_CASES)
def test_explicit_mismatching_resolution_warns(
    make_model: ModelFactory,
    make_stimulus: Callable[[float], Stimulus],
    parameters: dict[str, list[float]],
):
    """Test that an explicit resolution that differs from the frame duration warns rather than being overridden."""
    with pytest.warns(ResolutionMismatchWarning) as record:
        make_model(1.0)(make_stimulus(FRAME_DURATION), pd.DataFrame(parameters))

    # The CST model also warns for its two temporal channels
    assert any("'impulse_model' has a resolution of 1.0 s" in str(warning.message) for warning in record)


def test_single_precision_frame_duration_does_not_warn():
    """Test that a frame duration read from a single precision header still matches the explicit resolution."""
    model = _gaussian(float(np.float32(0.9)))

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionMismatchWarning)

        model.check_stimulus(_prf_stimulus(0.9))


def test_no_impulse_model_does_not_warn():
    """Test that a model without an impulse submodel accepts any frame duration."""
    model = Gaussian2DPRFModel(impulse_model=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionMismatchWarning)

        model.check_stimulus(_prf_stimulus(FRAME_DURATION))


@pytest.mark.parametrize("fitter_class", [GridFitter, LeastSquaresFitter, SGDFitter])
def test_fitter_warns_on_explicit_mismatching_resolution(fitter_class: type):
    """Test that fitters check the model against the stimulus once, when they are created."""
    with pytest.warns(ResolutionMismatchWarning, match="impulse_model"):
        fitter_class(_gaussian(1.0), _prf_stimulus(FRAME_DURATION))


@pytest.mark.parametrize("fitter_class", [GridFitter, LeastSquaresFitter, SGDFitter])
def test_fitter_does_not_warn_by_default(fitter_class: type):
    """Test that fitters accept a default model for any frame duration."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResolutionMismatchWarning)

        fitter_class(_gaussian(None), _prf_stimulus(FRAME_DURATION))
