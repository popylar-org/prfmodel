"""Validate prfmodel's 2D Gaussian pRF against prfpy.

Comparison strategy
-------------------
--- Non HRF Comparison ---
We compare the pre-HRF neural response: the dot product of the Gaussian RF
with the stimulus design matrix, summed over the spatial dimensions.

The pre-HRF comparison isolates the spatial encoding step,
the meaningful invariant between packages.

- prfmodel: Gaussian2DPRFModel(impulse_model=None, scaling_model=None)
  normalised RF: exp(-d^2/2*sigma^2) / (2*pi*sigma^2)
- prfpy:    gauss2D_iso_cart -> unnormalised RF: exp(-d^2/2*sigma^2)

--- Comparison with including HRF ---
Adds a second check that compares the full HRF-convolved prediction.  Both
sides are locked to the SPM canonical double-gamma (no time-derivative
component) by explicitly setting prfmodel's parameters to match nilearn's
_gamma_difference_hrf defaults:

  delay=6, dispersion=1, undershoot=16, u_dispersion=1, ratio=1/6, weight_deriv=0

Setting weight_deriv=0 reduces DerivativeTwoGammaImpulse to a plain
TwoGammaImpulse, so the two kernels are mathematically identical.

Two subtleties to be aware of when reading the prfpy-side implementation:

1. nilearn's spm_hrf(t_r, oversampling=50) returns the kernel on an
   *oversampled* grid (1600 points at 0.02 s for TR=1).  Using oversampling=1
   instead shifts the gamma evaluation by loc=dt=1 s, misaligning the peak by a
   full second.  We therefore fetch the oversampled kernel and downsample by
   slicing every 50th point, making the shift only 0.02 s (negligible).

2. prfmodel includes baseline and amplitude in its full prediction.  A non-zero
   baseline adds a constant offset that, after normalisation, changes the shape
   of the normalised timeseries.  We set baseline=0 and amplitude=1 in the
   SPM-params DataFrame so both sides return a raw convolved signal.

Coordinate conventions
----------------------
prfmodel stimulus grid: shape (H, W, 2), last axis = [y, x] in visual-angle degrees.
prfpy gauss2D_iso_cart(x, y, mu=(x_pos, y_pos), sigma):
  - first positional arg is the x-grid, second is the y-grid
  - mu[0] = x centre, mu[1] = y centre
"""

import sys
from collections.abc import Callable
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import spm_hrf
from scipy import signal

try:
    from prfpy.rf import gauss2D_iso_cart
except ImportError:
    print(
        "ERROR: prfpy is not installed.\n"
        "Install with: pip install prfpy\n"
        "or: pip install git+https://github.com/VU-Cog-Sci/prfpy.git",
    )
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import BASE_MODEL_PARAMS
from shared import RTOL
from shared import PRFStimulus
from shared import compare_predictions
from shared import load_stimulus
from shared import make_params
from shared import prfmodel_response

# SPM canonical double-gamma: weight_deriv=0 collapses DerivativeTwoGamma → TwoGamma.
# Parameters match nilearn's _gamma_difference_hrf defaults exactly:
# shape1=delay/dispersion=6, rate1=dispersion=1, shape2=undershoot/u_dispersion=16, rate2=1, c=ratio=1/6.
_SPM_HRF_PARAMS = {
    "delay": 6.0,
    "dispersion": 1.0,
    "undershoot": 16.0,
    "u_dispersion": 1.0,
    "ratio": 1 / 6,
    "weight_deriv": 0.0,
}

# Looser tolerance for the with-HRF check, set from the measured floor rather than chosen.
#
# The convolution is not the reason: prfmodel's Keras depthwise convolution and the scipy
# fftconvolve used on the prfpy side agree to 4e-08 of peak when handed the same kernel. Nor is
# prfmodel's kernel: an exact scipy kernel on prfmodel's own grid agrees to 2.9e-07. The residual is
# how nilearn lays out its time axis, and it has two parts that partly cancel: a `loc=dt` shift of
# one oversampled step (3.76e-03 on its own) and a `linspace` grid whose spacing is `32/31` s rather
# than 1 (7.13e-04 on its own). Downsampling removes neither. Together they measure 2.97e-03 of peak
# in the convolved prediction, and no change on the prfmodel side can close it.
#
# 5e-3 sits above that floor and well below a real defect: against the same centre-sampled
# reference, the class of error this check exists to catch measures 9.64e-02 when the kernel is
# mis-sampled by half a step and 1.88e-01 when it is off by a full one.
RTOL_WITH_HRF: float = 5e-3


def _prfpy_response(stimulus: PRFStimulus) -> np.ndarray:
    """Compute pre-HRF response using prfpy's gauss2D_iso_cart.

    Returns a 1-D array of length n_frames (unnormalised scale).
    """
    # Extract x and y coordinate grids from the prfmodel stimulus.
    # grid[:, :, 0] = y-axis, grid[:, :, 1] = x-axis (dimension_labels=['y', 'x'])
    y_grid = stimulus.grid[:, :, 0]  # shape (H, W)
    x_grid = stimulus.grid[:, :, 1]  # shape (H, W)

    # gauss2D_iso_cart returns exp(-((x-mu_x)^2 + (y-mu_y)^2) / (2*sigma^2)) - unnormalised.
    rf = gauss2D_iso_cart(
        x=x_grid,
        y=y_grid,
        mu=(BASE_MODEL_PARAMS["mu_x"], BASE_MODEL_PARAMS["mu_y"]),
        sigma=BASE_MODEL_PARAMS["sigma"],
    )  # (H, W)

    # Project RF onto stimulus: sum over spatial dims for each time frame.
    # design shape: (T, H, W)
    return np.einsum("thw,hw->t", stimulus.design, rf)  # (T,)


def _make_spm_params() -> pd.DataFrame:
    """Return params that lock prfmodel to the SPM canonical double-gamma HRF.

    baseline=0 and amplitude=1 so prfmodel returns the raw convolved signal with
    no constant offset — directly comparable to the prfpy side.
    """
    return pd.DataFrame(
        {
            "mu_x": [BASE_MODEL_PARAMS["mu_x"]],
            "mu_y": [BASE_MODEL_PARAMS["mu_y"]],
            "sigma": [BASE_MODEL_PARAMS["sigma"]],
            **{k: [v] for k, v in _SPM_HRF_PARAMS.items()},
            "baseline": [0.0],  # no baseline: prfpy does not support baseline
            "amplitude": [1.0],  # no amplitude: prfpy returns unscaled signal
        },
    )


def _prfpy_response_with_hrf(pre_hrf: np.ndarray) -> np.ndarray:
    """Convolve pre-HRF neural response with the SPM canonical HRF via scipy.

    nilearn's spm_hrf returns the kernel on an oversampled grid (default 50x).
    Calling it with oversampling=1 introduces a loc=dt=1 s shift that misaligns
    the kernel by a full second relative to prfmodel's gamma evaluation. Instead
    we oversample at the default rate and then downsample to 1 s/sample, so the
    loc shift is only 0.02 s and its effect is negligible.

    The downsample starts half a step in rather than at index 0, because prfmodel samples
    each frame at the centre of the interval it represents (0.5 s, 1.5 s, ...) while nilearn
    samples the leading edges. Reading both at the centres compares the two kernels rather
    than the two conventions: on a common grid they agree to 2.97e-03 of peak, against
    9.44e-02 when the grids sit half a step apart.

    TR=1 s is assumed — the resolution at which prfmodel samples its impulse kernel.
    """
    oversampling = 50
    kernel = spm_hrf(t_r=1.0, oversampling=oversampling)[oversampling // 2 :: oversampling]
    pad_len = len(kernel) - 1
    padded = np.pad(pre_hrf, (pad_len, 0), mode="edge")
    convolved = signal.fftconvolve(padded, kernel)
    return convolved[pad_len : pad_len + len(pre_hrf)]


def check_pre_hrf(stimulus: PRFStimulus) -> None:
    """Assert that prfmodel and prfpy agree on the spatial encoding step."""
    params = make_params()
    ref = prfmodel_response(stimulus, params, with_hrf=False)
    prfpy = _prfpy_response(stimulus)
    if not compare_predictions(ref, prfpy, "prfpy (pre-HRF)"):
        msg = (
            f"prfmodel and prfpy disagree on the pre-HRF response by more than {RTOL:.0e} of peak. "
            "This is the spatial encoding step: the Gaussian receptive field projected onto the "
            "stimulus design. A failure here means the receptive field, the stimulus grid "
            "convention, or the projection disagrees with prfpy."
        )
        raise AssertionError(msg)


def check_with_hrf(stimulus: PRFStimulus) -> None:
    """Assert that prfmodel and prfpy agree on the full prediction, HRF included.

    Together with ``check_pre_hrf`` this localises a disagreement: if the pre-HRF check passes and
    this one fails, the cause is the impulse response or the convolution. Both kernels are read at
    prfmodel's frame centres (see ``_prfpy_response_with_hrf``), on which they agree to 2.97e-03 of
    peak, the floor set by nilearn's time-axis layout; see ``RTOL_WITH_HRF``.

    """
    spm_params = _make_spm_params()
    ref_hrf = prfmodel_response(stimulus, spm_params, with_hrf=True)
    prfpy_hrf = _prfpy_response_with_hrf(_prfpy_response(stimulus))
    if not compare_predictions(ref_hrf, prfpy_hrf, "prfpy (with HRF)", rtol=RTOL_WITH_HRF):
        msg = (
            f"prfmodel and prfpy disagree on the full prediction by more than {RTOL_WITH_HRF:.0e} "
            "of peak. The pre-HRF check isolates whether the cause is spatial or temporal: if that "
            "one passes and this one fails, the discrepancy is in the impulse response or the "
            "convolution. See this function's docstring for the known cause."
        )
        raise AssertionError(msg)


def _run(check: Callable[[PRFStimulus], None], stimulus: PRFStimulus) -> str | None:
    """Run a single check, returning its failure message, or None if it passed."""
    try:
        check(stimulus)
    except AssertionError as exc:
        return str(exc)
    return None


def main() -> None:
    """Run prfpy comparisons and exit with 0 if all pass, 1 if any fail."""
    stimulus = load_stimulus()

    failures = [msg for msg in (_run(check, stimulus) for check in (check_pre_hrf, check_with_hrf)) if msg]
    for msg in failures:
        print(f"\n{msg}\n")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
