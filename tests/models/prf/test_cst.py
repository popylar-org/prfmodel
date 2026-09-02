"""Tests for the compressive spatiotemporal (CST) pRF model."""

import warnings
import numpy as np
import pandas as pd
import pytest
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import SustainedImpulse
from prfmodel.impulse import TransientImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.models.prf import Gaussian2DCSTPRFModel
from prfmodel.models.prf import Gaussian2DPRFTuning
from prfmodel.models.prf import PRFStimulusEncoder
from prfmodel.models.prf.canonical import ResolutionMismatchWarning
from prfmodel.scaling import Baseline
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup


class TestGaussian2DCSTPRFModel(PRFStimulusSetup):
    """Tests for the Gaussian2DCSTPRFModel class."""

    num_units = 3

    @pytest.fixture
    def prf_model(self):
        """CST pRF model object."""
        return Gaussian2DCSTPRFModel()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "n": [0.5, 0.8, 1.0],
                "amplitude_sustained": [1.0, 0.5, 2.0],
                "amplitude_transient": [0.5, 1.0, 0.0],
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 1.5, 2.0],
                "time_to_peak": [4.0, 5.0, 6.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [-0.5, -0.5, -0.5],
                "baseline": [-0.5, 0.5, 2.0],
            },
        )

    def test_parameter_names(self, prf_model: Gaussian2DCSTPRFModel):
        """Test that parameter names aggregate the submodels plus the CST-specific parameters."""
        expected = [
            "n",
            "amplitude_sustained",
            "amplitude_transient",
            "mu_y",
            "mu_x",
            "sigma",
            "time_to_peak",
            *DerivativeTwoGammaImpulse().parameter_names,
            *Baseline().parameter_names,
        ]

        assert prf_model.parameter_names == expected

    def test_time_to_peak_is_shared_across_channels(self, prf_model: Gaussian2DCSTPRFModel):
        """Test that the sustained and transient channels contribute a single `time_to_peak` column."""
        assert prf_model.parameter_names.count("time_to_peak") == 1

    def test_call_shape(self, prf_model: Gaussian2DCSTPRFModel, stimulus: PRFStimulus, params: pd.DataFrame):
        """Test that the predicted response has one value per unit per stimulus frame."""
        resp = prf_model(stimulus, params)

        assert resp.shape == (self.num_units, stimulus.design.shape[0])

    def test_zero_betas_leave_only_the_baseline(
        self,
        prf_model: Gaussian2DCSTPRFModel,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that both channels are switched off by their weights, leaving the scaling model's baseline."""
        params = params.assign(amplitude_sustained=0.0, amplitude_transient=0.0)

        resp = np.asarray(prf_model(stimulus, params))

        assert resp == pytest.approx(np.repeat(params[["baseline"]].to_numpy(), resp.shape[1], axis=1))

    def test_transient_weight_changes_the_prediction(
        self,
        prf_model: Gaussian2DCSTPRFModel,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that the transient channel contributes to the response rather than being discarded."""
        resp_with = np.asarray(prf_model(stimulus, params.assign(amplitude_transient=1.0)))
        resp_without = np.asarray(prf_model(stimulus, params.assign(amplitude_transient=0.0)))

        assert not np.allclose(resp_with, resp_without)

    def test_transient_channels_rectify_to_the_absolute_response(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that the on- and off-transient channels rectify opposite halves of the same response.

        With `n=1` the two channels are `ReLU(r)` and `ReLU(-r)`, which sum to `abs(r)`. The impulse model is
        switched off so that the biphasic impulse response cannot reintroduce negative values downstream.

        """
        prf_model = Gaussian2DCSTPRFModel(impulse_model=None)
        params = params.assign(amplitude_sustained=0.0, amplitude_transient=1.0, n=1.0, baseline=0.0)

        resp = np.asarray(prf_model(stimulus, params))

        assert np.all(resp >= 0.0)
        # The transient response is not trivially flat at the rectifier floor
        assert resp.max() > 1.0

    def test_sustained_channel_is_non_negative_without_an_impulse_model(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that the rectified and compressed sustained channel never contributes a negative response."""
        prf_model = Gaussian2DCSTPRFModel(impulse_model=None)
        params = params.assign(amplitude_sustained=1.0, amplitude_transient=0.0, baseline=0.0)

        resp = np.asarray(prf_model(stimulus, params))

        assert np.all(resp >= 0.0)

    def test_transient_channels_sum_to_the_absolute_channel_response(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that the on- and off-transient channels are the two rectified halves of one convolution.

        Builds the transient channel response independently from the same submodels and asserts that the
        model's prediction is its absolute value. This pins the wiring: an off-transient built from anything
        other than the negated on-transient, or a channel convolved with the wrong kernel, breaks it.

        The tolerances are loose for two independent reasons. The rectifier floors at `min_response` rather
        than zero, so the prediction is `abs(channel) + min_response` and exactly `2 * min_response` where the
        channel response is zero, which the absolute tolerance absorbs. The relative tolerance is set for
        cross-backend portability: the two paths run the same convolution but reach it differently, and the
        torch backend does not preserve float64 through it, so agreement is only to roughly single precision.
        Both are far tighter than the errors this test exists to catch, which are order-one.

        """
        prf_model = Gaussian2DCSTPRFModel(impulse_model=None)
        params = params.assign(amplitude_sustained=0.0, amplitude_transient=1.0, n=1.0, baseline=0.0)

        encoded = PRFStimulusEncoder()(
            stimulus,
            Gaussian2DPRFTuning()(stimulus, params, dtype="float64"),
            params,
            dtype="float64",
        )
        channel = convolve_prf_impulse_response(
            encoded,
            TransientImpulse()(params, dtype="float64"),
            dtype="float64",
        )

        observed = np.asarray(prf_model(stimulus, params, dtype="float64"))

        np.testing.assert_allclose(observed, np.abs(np.asarray(channel)), rtol=1e-6, atol=1e-9)

    def test_exponent_compresses_the_response(
        self,
        prf_model: Gaussian2DCSTPRFModel,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that `n` is applied as an exponent rather than ignored."""
        resp_linear = np.asarray(prf_model(stimulus, params.assign(n=1.0)))
        resp_compressed = np.asarray(prf_model(stimulus, params.assign(n=0.5)))

        assert not np.allclose(resp_linear, resp_compressed)

    def test_submodels_are_the_cst_channels(self, prf_model: Gaussian2DCSTPRFModel):
        """Test that the model is wired up with the sustained and transient channel impulse models."""
        assert isinstance(prf_model.models["sustained_model"], SustainedImpulse)
        assert isinstance(prf_model.models["transient_model"], TransientImpulse)


class TestCSTPRFModelValidation:
    """Tests for the submodel consistency checks in `CSTPRFModel.__init__`."""

    @pytest.mark.parametrize("channel", ["sustained_model", "transient_model"])
    def test_channel_model_cannot_be_none(self, channel: str):
        """Test that a missing channel model is rejected rather than failing at prediction time.

        The model is defined by its three channels, and a channel is silenced by setting its weight to zero,
        so `None` has no meaning here. Without this check the model constructs, reports a plausible
        `parameter_names`, and only fails inside `__call__` with an unrelated `TypeError`.

        """
        kwargs: dict[str, None] = {channel: None}

        with pytest.raises(ValueError, match="required"):
            Gaussian2DCSTPRFModel(**kwargs)  # type: ignore[arg-type]  # the None is the point of the test

    def test_channel_resolution_mismatched_with_impulse_model_warns(self):
        """Test that channels sampled differently from the impulse response are reported.

        `convolve_prf_impulse_response` treats matched sampling as an unchecked precondition, so a mismatch
        silently produces a wrong prediction rather than an error.

        """
        with pytest.warns(ResolutionMismatchWarning, match="resolution"):
            Gaussian2DCSTPRFModel(
                sustained_model=SustainedImpulse(resolution=1.5),
                transient_model=TransientImpulse(resolution=1.5),
            )

    def test_channels_mismatched_with_each_other_warn(self):
        """Test that the two channels sampled differently from one another are reported."""
        with pytest.warns(ResolutionMismatchWarning, match="resolution"):
            Gaussian2DCSTPRFModel(sustained_model=SustainedImpulse(resolution=1.5))

    def test_matching_resolutions_do_not_warn(self):
        """Test that the default configuration, where every time axis agrees, is silent."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            Gaussian2DCSTPRFModel()

    def test_no_impulse_model_compares_only_the_channels(self):
        """Test that the impulse model is excluded from the comparison when it is not used."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            Gaussian2DCSTPRFModel(
                impulse_model=None,
                sustained_model=SustainedImpulse(resolution=1.5),
                transient_model=TransientImpulse(resolution=1.5),
            )
