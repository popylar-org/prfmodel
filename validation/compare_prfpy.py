"""Validate prfmodel's 2D Gaussian pRF against prfpy.

Comparison strategy
-------------------
--- Non HRF Comparison ---
We compare the pre-HRF neural response: the dot product of the Gaussian RF
with the stimulus design matrix, summed over the spatial dimensions.

The pre-HRF comparison isolates the spatial encoding step,
the meaningful invariant between packages.

- prfmodel: Gaussian2DPRFModel(impulse_model=None, temporal_model=None)
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

# Looser tolerance for the with-HRF check: different convolution backends
# (prfmodel uses Keras depthwise conv, prfpy side uses scipy fftconvolve) and
# the residual 0.02 s loc shift from the oversampling approach introduce small
# numerical differences not present in the pre-HRF comparison.
RTOL_WITH_HRF: float = 1e-3


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
    we oversample at the default rate and then downsample to 1 s/sample (every 50th
    point), so the loc shift is only 0.02 s and its effect is negligible.

    TR=1 s is assumed — the resolution at which prfmodel samples its impulse kernel.
    """
    oversampling = 50
    kernel = spm_hrf(t_r=1.0, oversampling=oversampling)[::oversampling]
    pad_len = len(kernel) - 1
    padded = np.pad(pre_hrf, (pad_len, 0), mode="edge")
    convolved = signal.fftconvolve(padded, kernel)
    return convolved[pad_len : pad_len + len(pre_hrf)]


def main() -> None:
    """Run prfpy comparisons and exit with 0 if all pass, 1 if any fail."""
    stimulus = load_stimulus()

    # Pre-HRF check: spatial encoding only.
    params = make_params()
    ref = prfmodel_response(stimulus, params, with_hrf=False)
    prfpy = _prfpy_response(stimulus)
    passed_pre = compare_predictions(ref, prfpy, "prfpy (pre-HRF)")

    # Full-prediction check: spatial encoding + SPM canonical HRF convolution.
    spm_params = _make_spm_params()
    ref_hrf = prfmodel_response(stimulus, spm_params, with_hrf=True)
    prfpy_hrf = _prfpy_response_with_hrf(_prfpy_response(stimulus))
    passed_hrf = compare_predictions(ref_hrf, prfpy_hrf, "prfpy (with HRF)", rtol=RTOL_WITH_HRF)

    sys.exit(0 if passed_pre and passed_hrf else 1)


if __name__ == "__main__":
    main()
