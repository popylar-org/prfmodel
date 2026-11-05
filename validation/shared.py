"""Shared fixtures and utilities for cross-package pRF validation.

Stimulus and parameters mirror the diff_gaussians tutorial.

HRF comparison is by default excluded from cross-package checks, because each
package implements its own HRF (nilearn SPM canonical, TDM lookup table, custom
double-gamma) with parameters that do not map directly across implementations.
Comparing the pre-HRF neural response — the dot product of the Gaussian RF with
the stimulus design matrix — isolates the spatial encoding step and makes the
comparison package-agnostic: if the RF and the projection agree, any remaining
differences in the full prediction must come from the temporal model.

Comparison criterion
--------------------
Both series are normalized to unit absolute sum before comparison, because packages
disagree on receptive-field scale. That normalization divides out `amplitude` and
`baseline`, so these checks constrain the *shape* of the prediction only and are
structurally unable to detect a scaling bug; scaling is covered by the in-repository
test suite instead.

Agreement is then measured as the maximum absolute difference relative to the peak of
the reference, which is the scale-free quantity for a timeseries. That same number is
the one printed and the one compared against the tolerance.
"""

import os
import numpy as np
import pandas as pd

# Must be set before importing prfmodel (selects the Keras backend).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus

BASE_MODEL_PARAMS: dict = {
    "mu_x": -2.1,
    "mu_y": 1.45,
    "sigma": 1.35,
    "amplitude": 1.2,
    "baseline": 10.0,
}

HRF_PARAMS: dict = {
    "delay": 6.0,
    "dispersion": 0.9,
    "undershoot": 12.0,
    "u_dispersion": 0.9,
    "ratio": 0.48,
    "weight_deriv": -0.5,
}

# Tolerance for the normalised timeseries comparison, as a fraction of the reference peak.
RTOL: float = 1e-4


def load_stimulus() -> PRFStimulus:
    """Return the standard bar stimulus (design: 170 x 128 x 128, x and y spanning +/-4.007 deg)."""
    stimulus = load_2d_prf_bar_stimulus()
    if isinstance(stimulus, tuple):  # pragma: no cover - defensive, the default returns a single stimulus
        stimulus = stimulus[0]
    return stimulus


def make_params() -> pd.DataFrame:
    """Return a one-row DataFrame with pre-chosen parameters."""
    return pd.DataFrame(
        {
            "mu_x": [BASE_MODEL_PARAMS["mu_x"]],
            "mu_y": [BASE_MODEL_PARAMS["mu_y"]],
            "sigma": [BASE_MODEL_PARAMS["sigma"]],
            **{k: [v] for k, v in HRF_PARAMS.items()},
            "baseline": [BASE_MODEL_PARAMS["baseline"]],
            "amplitude": [BASE_MODEL_PARAMS["amplitude"]],
        },
    )


def prfmodel_response(stimulus: PRFStimulus, params: pd.DataFrame, *, with_hrf: bool = False) -> np.ndarray:
    """Return prfmodel's 2D Gaussian pRF prediction as a 1-D NumPy array.

    When with_hrf=False (default) the impulse and temporal models are disabled,
    returning only the normalised Gaussian RF projected onto the stimulus design.
    """
    model = Gaussian2DPRFModel() if with_hrf else Gaussian2DPRFModel(impulse_model=None, scaling_model=None)
    return np.asarray(model(stimulus, params)).ravel()


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1-D array to unit absolute sum."""
    arr = np.asarray(arr, dtype=float).ravel()
    total = np.abs(arr).sum()
    return arr / total if total > 0 else arr


def peak_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return the maximum absolute difference between two series, as a fraction of b's peak.

    Both series are normalized to unit absolute sum first, so this is scale-free: it answers
    "how far apart are these two timeseries, measured against the size of the reference?".
    """
    a_norm, b_norm = normalize(a), normalize(b)
    peak = float(np.abs(b_norm).max())
    if peak == 0.0:
        msg = "Reference series is all zeros; cannot compute a peak-relative error."
        raise ValueError(msg)
    return float(np.abs(a_norm - b_norm).max()) / peak


def compare_predictions(a: np.ndarray, b: np.ndarray, package: str, *, rtol: float = RTOL) -> bool:
    """Compare two normalised timeseries; print the result and return True (pass) or False (fail).

    Returns bool rather than calling sys.exit so that callers running multiple checks can
    collect all results before deciding the final exit code. Callers are expected to assert
    on the returned value.

    Both arrays are normalized to unit absolute sum before comparison, to handle RF scale
    differences across packages. See the module docstring for what that normalization means
    for the interpretation of a pass.
    """
    error = peak_relative_error(a, b)
    passed = error <= rtol
    label = f"prfmodel vs {package} (normalised)"
    status = "PASS" if passed else "FAIL"
    print(f"{status}  {label}  peak_rel_error={error:.2e}  tol={rtol}")
    return passed
