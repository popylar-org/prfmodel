"""Tests that pin the derived pRF models against known-exact answers.

The suite's numerical assertions for these models are regression snapshots, which lock in whatever
the code produces and so cannot tell a correct composition from a self-consistent wrong one. The
cross-package checks in `validation/` do not cover them either: they only compare the 2D Gaussian,
and they normalize both series to unit absolute sum, which divides `amplitude` and `baseline` out.

Each test below drives a derived model to a degenerate configuration in which it must reduce, exactly
or to a stated tolerance, to a model that *is* externally anchored -- `Gaussian2DPRFModel`, pinned
against `scipy.stats.multivariate_normal` in `tests/models/prf/test_gaussian.py` and against prfpy
and braincoder in `validation/`. Because the reduction runs the whole pipeline -- receptive field,
stimulus encoding, HRF convolution, scaling -- it constrains the composition, not just the parts.

"""

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy import spatial
from prfmodel.models.cf._gaussian import GaussianCFResponse
from prfmodel.models.prf import DivNormGaussian2DPRFModel
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import Gaussian2DCSSPRFModel
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.models.prf._gaussian import predict_gaussian_response
from prfmodel.stimuli import CFStimulus
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup

HRF_PARAMS: dict[str, list[float]] = {
    "delay": [6.0, 5.0],
    "dispersion": [1.0, 0.9],
    "undershoot": [16.0, 12.0],
    "u_dispersion": [1.0, 0.9],
    "ratio": [1 / 6, 0.48],
    "weight_deriv": [0.0, 0.5],
}

MU_Y: list[float] = [1.0, -0.5]
MU_X: list[float] = [-1.0, 0.5]
SIGMA: list[float] = [1.0, 1.5]
BASELINE: list[float] = [0.1, -0.3]
AMPLITUDE: list[float] = [1.2, -2.0]


def _gaussian_params() -> pd.DataFrame:
    """Return reference parameters for `Gaussian2DPRFModel`, two units."""
    return pd.DataFrame(
        {
            "mu_y": MU_Y,
            "mu_x": MU_X,
            "sigma": SIGMA,
            **HRF_PARAMS,
            "baseline": BASELINE,
            "amplitude": AMPLITUDE,
        },
    )


class TestReductionToGaussian(PRFStimulusSetup):
    """Tests that each derived model reduces to the plain Gaussian when its extra stage is disabled.

    The reduction is asserted on the full predicted timeseries, so a stage applied on the wrong
    axis or built from the wrong parameters breaks it. It does not constrain where the *linear*
    stages sit relative to the convolution: a unit-sum kernel with edge padding leaves
    `conv(a * r + b) == a * conv(r) + b`, so moving the scaling across the convolution is
    invisible here. Only the nonlinear stages are pinned in position.

    """

    @pytest.fixture
    def reference(self, stimulus: PRFStimulus) -> np.ndarray:
        """Return the prediction of the externally anchored `Gaussian2DPRFModel`."""
        return np.asarray(Gaussian2DPRFModel()(stimulus, _gaussian_params(), dtype="float64"))

    def test_dog_with_zero_surround_is_gaussian(self, stimulus: PRFStimulus, reference: np.ndarray):
        """Test that a difference of Gaussians with no surround is a plain Gaussian.

        `DoG2DPRFModel` subtracts `amplitude_surround` times the surround response from
        `amplitude_center` times the center response. At `amplitude_surround=0` the surround drops
        out exactly and the model must reproduce the Gaussian, whatever `sigma_surround` is.

        """
        params = pd.DataFrame(
            {
                "mu_y": MU_Y,
                "mu_x": MU_X,
                "sigma_center": SIGMA,
                # Deliberately not the center sigma: a surround that leaked through would show up.
                "sigma_surround": [3.0, 4.0],
                "amplitude_center": AMPLITUDE,
                "amplitude_surround": [0.0, 0.0],
                **HRF_PARAMS,
                "baseline": BASELINE,
            },
        )

        observed = np.asarray(DoG2DPRFModel()(stimulus, params, dtype="float64"))

        np.testing.assert_allclose(observed, reference, rtol=1e-10)

    def test_css_with_unit_exponent_is_gaussian(self, stimulus: PRFStimulus, reference: np.ndarray):
        """Test that compressive spatial summation with `n=1, gain=1` is a plain Gaussian.

        `CompressiveEncoder` computes `gain * maximum(response, min_response) ** n`, so the identity
        holds only up to the `min_response` floor, which replaces an exactly-zero encoded response
        with 1e-10. That floor is the reason for the absolute tolerance below; it bites on frames
        where the bar does not overlap the receptive field at all.

        """
        params = pd.DataFrame(
            {
                "mu_y": MU_Y,
                "mu_x": MU_X,
                "sigma": SIGMA,
                "gain": [1.0, 1.0],
                "n": [1.0, 1.0],
                **HRF_PARAMS,
                "baseline": BASELINE,
                "amplitude": AMPLITUDE,
            },
        )

        observed = np.asarray(Gaussian2DCSSPRFModel()(stimulus, params, dtype="float64"))

        np.testing.assert_allclose(observed, reference, rtol=1e-8, atol=1e-9)

    def test_css_gain_scales_each_unit_by_its_own_value(self, stimulus: PRFStimulus):
        """Test that at `n=1` the CSS model is a Gaussian whose amplitude is scaled by `gain`.

        `gain` differs across units here, unlike in `test_css_with_unit_exponent_is_gaussian`. With
        `gain=1` for every unit the reduction cannot see the unit axis at all: reversing `gain` and
        `n` across units leaves the result bit-identical. Distinct per-unit values pin the axis.

        """
        gain = [2.0, 3.0]

        params = pd.DataFrame(
            {
                "mu_y": MU_Y,
                "mu_x": MU_X,
                "sigma": SIGMA,
                "gain": gain,
                "n": [1.0, 1.0],
                **HRF_PARAMS,
                "baseline": BASELINE,
                "amplitude": AMPLITUDE,
            },
        )

        # At n=1 the encoder is a plain multiplication, so `gain` folds into `amplitude`.
        expected_params = _gaussian_params()
        expected_params["amplitude"] = [a * g for a, g in zip(AMPLITUDE, gain, strict=True)]
        expected = np.asarray(Gaussian2DPRFModel()(stimulus, expected_params, dtype="float64"))

        observed = np.asarray(Gaussian2DCSSPRFModel()(stimulus, params, dtype="float64"))

        np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-9)

    def test_div_norm_without_normalization_is_gaussian(self, stimulus: PRFStimulus, reference: np.ndarray):
        """Test that divisive normalization with an inert denominator is a plain Gaussian.

        The model computes `(a * R_act + b) / (c * R_norm + d) - b / d`. Setting `c=0` and `d=1`
        makes the denominator identically 1 and cancels the `b` terms, leaving `a * R_act` -- so the
        result must match the Gaussian scaled by `a`, for any `b`.

        `DivNormGaussian2DPRFModel` scales with `Baseline` rather than `BaselineAmplitude`, applying
        `a` before the convolution instead of after. The identity holds regardless because
        convolution is linear, which is itself part of what this test pins.

        """
        params = pd.DataFrame(
            {
                "mu_y": MU_Y,
                "mu_x": MU_X,
                "sigma_activation": SIGMA,
                # Inert, so its value must not matter.
                "sigma_normalization": [2.0, 2.5],
                "amplitude_activation": AMPLITUDE,
                "amplitude_normalization": [0.0, 0.0],
                "baseline_activation": [0.7, -0.4],
                "baseline_normalization": [1.0, 1.0],
                **HRF_PARAMS,
                "baseline": BASELINE,
            },
        )

        observed = np.asarray(DivNormGaussian2DPRFModel()(stimulus, params, dtype="float64"))

        np.testing.assert_allclose(observed, reference, rtol=1e-8, atol=1e-9)


class TestGaussianNormalization:
    """Tests the normalization constant of the Gaussian receptive field.

    `predict_gaussian_response` is already checked against `scipy.stats.multivariate_normal.pdf`
    elsewhere. What that comparison cannot express is *why* the constant is what it is, which is what
    makes an analogous error in another module hard to spot -- see the CF case below.

    """

    def test_2d_gaussian_integrates_to_one(self):
        """Test that the 2D receptive field is a probability density over the visual field."""
        sigma = 1.0
        half_width = 8.0 * sigma  # Wide enough that the truncated tail is far below the tolerance.
        num_points = 401

        axis = np.linspace(-half_width, half_width, num_points)
        yv, xv = np.meshgrid(axis, axis, indexing="ij")
        grid = np.stack((yv, xv), axis=-1)

        response = np.asarray(
            predict_gaussian_response(grid, np.array([[0.0, 0.0]]), np.array([[sigma]])),
        )[0]

        integral = integrate.simpson(integrate.simpson(response, x=axis), x=axis)

        assert integral == pytest.approx(1.0, abs=1e-6)

    def test_cf_gaussian_integrates_to_one_over_a_flat_surface(self):
        """Test that the connective-field Gaussian is a density over a two-dimensional surface.

        The CF response is defined on a distance matrix rather than a coordinate grid, so it carries
        no explicit dimensionality and its normalizer cannot be checked by inspection. Laying the
        vertices out on a regular planar patch supplies the missing area element: summing the
        response over vertices and multiplying by the area each vertex occupies must give 1 if, and
        only if, the normalizer is the two-dimensional `2 * pi * sigma**2`. The one-dimensional
        `sqrt(2 * pi * sigma**2)` would leave the integral off by a factor of `sqrt(2 * pi) * sigma`.

        A plane is the right test surface even though real cortex is curved: the normalizer is a
        local quantity, and any smooth surface is locally flat.

        """
        sigma = 1.0
        half_width = 6.0 * sigma
        num_per_side = 31

        axis = np.linspace(-half_width, half_width, num_per_side)
        spacing = float(axis[1] - axis[0])
        yv, xv = np.meshgrid(axis, axis, indexing="ij")
        vertices = np.stack((yv.ravel(), xv.ravel()), axis=-1)

        distance_matrix = spatial.distance.cdist(vertices, vertices)
        num_vertices = distance_matrix.shape[0]
        stimulus = CFStimulus(
            distance_matrix=distance_matrix,
            source_response=np.zeros((num_vertices, 4)),
        )

        centre = int(np.argmin(np.linalg.norm(vertices, axis=1)))
        params = pd.DataFrame({"center_index": [centre], "sigma": [sigma]})

        response = np.asarray(GaussianCFResponse()(stimulus, params, dtype="float64"))[0]

        # Rectangle rule: each vertex stands for a square patch of side `spacing`. It converges
        # very quickly for a Gaussian, so the tolerance is dominated by the truncated tail.
        integral = response.sum() * spacing**2

        assert integral == pytest.approx(1.0, abs=1e-4)
