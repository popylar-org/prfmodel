"""Tests that pin the HRF parameterization against external reference implementations.

The impulse models are otherwise covered by regression snapshots, which lock in whatever the code
currently produces and so cannot detect a parameterization that is self-consistent but does not match
the literature. These tests compare against `scipy.stats.gamma` and against nilearn, which define the
convention `delay` and `dispersion` are meant to follow.

"""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from scipy import stats
from prfmodel.impulse import ShiftedGammaImpulse
from prfmodel.impulse import TwoGammaImpulse
from prfmodel.impulse.defaults import default_two_gamma_impulse_glover_hrf
from prfmodel.impulse.defaults import default_two_gamma_impulse_spm_hrf

DURATION = 32.0


def _normalize(response: np.ndarray) -> np.ndarray:
    """Scale to unit peak so the comparison is about shape, not about the normalization convention."""
    return response / np.abs(response).max()


class TestGammaScaleConvention:
    """Tests that `dispersion` is the gamma scale, as in SPM, nilearn and Glover."""

    @pytest.mark.parametrize(("delay", "dispersion"), [(6.0, 0.9), (6.0, 1.0), (12.0, 0.9), (4.0, 2.0)])
    def test_single_gamma_component_matches_scipy(self, delay: float, dispersion: float):
        """Test that a single gamma component equals `scipy.stats.gamma.pdf` with `scale=dispersion`.

        The undershoot is switched off with `ratio=0`, leaving the positive component alone. This is
        the assertion that pins the parameterization: passing `dispersion` as a rate instead of a
        scale would need `scale=1/dispersion` here.

        """
        model = TwoGammaImpulse(duration=DURATION, resolution=1.0, norm=None, default_parameters=None)
        frames = np.asarray(model.frames)[0]

        parameters = pd.DataFrame(
            {
                "delay": [delay],
                "dispersion": [dispersion],
                "undershoot": [12.0],
                "u_dispersion": [1.0],
                "ratio": [0.0],
            },
        )

        observed = np.asarray(model(parameters, dtype="float64"))[0]
        expected = stats.gamma.pdf(frames, a=delay / dispersion, scale=dispersion)

        np.testing.assert_allclose(
            observed,
            expected,
            # The density is evaluated in log space, which loses a little accuracy far out in the
            # tail.
            rtol=1e-5,
            atol=1e-12,
            err_msg="'dispersion' is not being used as the gamma scale",
        )

    @pytest.mark.parametrize(("delay", "dispersion"), [(6.0, 0.9), (12.0, 0.9), (4.0, 2.0)])
    def test_delay_is_the_mean_of_the_component(self, delay: float, dispersion: float):
        """Test that `delay` is the mean of the gamma component in seconds.

        This is the property that makes `delay` interpretable and comparable across software; it holds
        only when `dispersion` is the scale, since the mean is ``(delay / dispersion) * dispersion``.

        """
        # A fine grid so the numerical integral is accurate out into the tail
        model = TwoGammaImpulse(duration=120.0, resolution=0.01, norm=None, default_parameters=None)
        frames = np.asarray(model.frames)[0]

        parameters = pd.DataFrame(
            {
                "delay": [delay],
                "dispersion": [dispersion],
                "undershoot": [12.0],
                "u_dispersion": [1.0],
                "ratio": [0.0],
            },
        )

        density = np.asarray(model(parameters, dtype="float64"))[0]
        # The frames are evenly spaced and the density has decayed to ~0 at both ends, so a weighted
        # average over the samples is an accurate estimate of the mean (and is numpy-version agnostic)
        mean = float(np.sum(frames * density) / np.sum(density))

        assert mean == pytest.approx(delay, rel=1e-3), f"Mean of the gamma component is {mean}, expected {delay}"

    def test_shifted_gamma_uses_the_same_convention(self):
        """Test that `ShiftedGammaImpulse` interprets `dispersion` the same way."""
        delay, dispersion, shift = 6.0, 0.9, 2.0

        model = ShiftedGammaImpulse(duration=DURATION, resolution=1.0, norm=None)
        frames = np.asarray(model.frames)[0]

        parameters = pd.DataFrame({"delay": [delay], "dispersion": [dispersion], "shift": [shift]})

        observed = np.asarray(model(parameters, dtype="float64"))[0]
        expected = stats.gamma.pdf(frames, a=delay / dispersion, loc=shift, scale=dispersion)

        np.testing.assert_allclose(
            observed,
            expected,
            # The density is evaluated in log space, which loses a little accuracy far out in the
            # tail. This is still around five orders of magnitude tighter than the discrepancy a
            # rate/scale mix-up produces, which is of order 1 in relative terms.
            rtol=1e-5,
            atol=1e-12,
            err_msg="ShiftedGammaImpulse does not use 'dispersion' as the gamma scale",
        )


class TestDefaultsMatchNilearn:
    """Tests that the packaged default parameter sets reproduce the HRFs they are named after."""

    # nilearn evaluates its gammas with loc=t_r/oversampling, a sub-sample shift prfmodel does not
    # apply, so the two agree closely rather than exactly. The tolerances below are far tighter than
    # the discrepancy a wrong parameterization produces (correlation 0.86, peak difference 1.2 s).
    oversampling: int = 125
    min_correlation: float = 0.999
    max_peak_difference_seconds: float = 0.1

    @pytest.mark.parametrize(
        ("name", "reference_fun", "defaults_fun"),
        [
            ("glover_hrf", glover_hrf, default_two_gamma_impulse_glover_hrf),
            ("spm_hrf", spm_hrf, default_two_gamma_impulse_spm_hrf),
        ],
    )
    def test_default_hrf_matches_nilearn(self, name: str, reference_fun: callable, defaults_fun: callable):
        """Test that the default parameter sets reproduce nilearn's canonical HRFs."""
        reference = reference_fun(1.0, oversampling=self.oversampling, time_length=DURATION)

        model = TwoGammaImpulse(
            duration=DURATION,
            resolution=DURATION / len(reference),
            norm="sum",
            default_parameters=name,
        )

        # The named default set and the function that produces it must agree
        assert model.default_parameters == defaults_fun()
        observed = np.asarray(model(pd.DataFrame(index=range(1)), dtype="float64"))[0]

        num_samples = min(len(observed), len(reference))
        observed, reference = _normalize(observed[:num_samples]), _normalize(reference[:num_samples])

        correlation = np.corrcoef(observed, reference)[0, 1]
        assert correlation > self.min_correlation, f"{name} correlates {correlation:.4f} with nilearn's {name}"

        seconds_per_sample = DURATION / num_samples
        peak_difference = abs(np.argmax(observed) - np.argmax(reference)) * seconds_per_sample
        assert peak_difference < self.max_peak_difference_seconds, (
            f"{name} peaks {peak_difference:.2f}s away from nilearn's {name}"
        )

    def test_defaults_are_documented_as_scale_parameters(self):
        """Test that the default dispersions are the nilearn values, not their reciprocals.

        The two conventions coincide when the dispersion is 1.0, which is why the SPM set alone cannot
        detect a rate/scale mix-up. The Glover set can, so it is checked explicitly.

        """
        assert default_two_gamma_impulse_glover_hrf()["dispersion"] == pytest.approx(0.9)
        assert default_two_gamma_impulse_glover_hrf()["u_dispersion"] == pytest.approx(0.9)


class TestKernelSampleSpacing:
    """Tests that the impulse kernel is sampled at the requested `resolution`.

    `resolution` is seconds per sample, so consecutive frames must be exactly that far apart. These
    tests assert at the resolution a user actually gets; `TestDefaultsMatchNilearn` compares on a
    heavily oversampled grid, where a spacing error is too small to see.

    """

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("resolution", [1.0, 1.5, 0.5])
    def test_frame_spacing_equals_resolution(self, resolution: float, dtype: str):
        """Test that consecutive impulse frames are exactly `resolution` seconds apart."""
        model = TwoGammaImpulse(duration=DURATION, resolution=resolution)
        frames = np.asarray(model.get_frames(dtype)).ravel()

        spacing = np.diff(frames)

        # float32 resolves the axis to ~1e-6 relative; float64 to machine precision.
        rtol = 1e-5 if dtype == "float32" else 1e-12
        np.testing.assert_allclose(spacing, resolution, rtol=rtol)

    def test_frames_are_built_at_the_requested_precision(self):
        """Test that a float64 axis carries float64 time values, not widened float32 ones.

        Both dtypes are compared against the exact arithmetic grid, which distinguishes an axis built
        at double precision from one built at single precision and cast up.

        """
        double_precision_limit = 1e-12
        single_precision_floor = 1e-9

        model = TwoGammaImpulse(duration=DURATION, resolution=0.5)
        exact = np.arange(model.num_frames) * 0.5 + model.offset

        as_32 = np.asarray(model.get_frames("float32")).ravel()
        as_64 = np.asarray(model.get_frames("float64")).ravel()

        assert np.abs(as_64 - exact).max() < double_precision_limit
        # And the float32 axis is genuinely coarser, so the assertion above has content.
        assert np.abs(as_32 - exact).max() > single_precision_floor

    def test_kernel_matches_nilearn_at_native_resolution(self):
        """Test that the default kernel reproduces nilearn's `spm_hrf` at 1 s sampling.

        `TestDefaultsMatchNilearn` establishes this on a finely oversampled grid; here it is asserted
        at the 1 s resolution a user gets by default. nilearn is evaluated oversampled and then
        downsampled, because `oversampling=1` applies a `loc=1 s` shift unrelated to what is tested.

        """
        oversampling = 100
        reference = spm_hrf(t_r=1.0, oversampling=oversampling, time_length=DURATION)[::oversampling]

        model = TwoGammaImpulse(duration=DURATION, resolution=1.0, norm="sum", default_parameters="spm_hrf")
        observed = np.asarray(model(pd.DataFrame(index=range(1)), dtype="float64"))[0]

        num_samples = min(len(observed), len(reference))
        observed, reference = _normalize(observed[:num_samples]), _normalize(reference[:num_samples])

        # Tolerance sits between the two regimes: the correctly sampled kernel agrees to ~6e-3 of
        # peak (nilearn's own time-axis layout, see validation/compare_prfpy.py), a kernel
        # mis-sampled by one step is off by ~6e-2.
        np.testing.assert_allclose(observed, reference, atol=2e-2)
