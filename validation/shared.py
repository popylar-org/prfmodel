"""Shared fixtures and utilities for cross-package pRF validation.

Stimulus and parameters mirror the diff_gaussians tutorial exactly.
The reference prediction uses prfmodel's Gaussian2DPRFModel without HRF or
temporal scaling, isolating the spatial encoding step (RF x stimulus design).
This makes the comparison package-agnostic: HRF implementations differ across
packages, but the Gaussian encoding step should produce identical timeseries.
"""

import os
import sys
import numpy as np
import pandas as pd

# Must be set before importing prfmodel (selects the Keras backend).
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.models import Gaussian2DPRFModel

# ---------------------------------------------------------------------------
# Canonical parameters - from the diff_gaussians tutorial.
# sigma maps to sigma_center (the Gaussian centre width).
# ---------------------------------------------------------------------------
MU_X: float = -2.1
MU_Y: float = 1.45
SIGMA: float = 1.35
AMPLITUDE: float = 1.2
BASELINE: float = 10.0

HRF_PARAMS: dict = {
    "delay": 6.0,
    "dispersion": 0.9,
    "undershoot": 12.0,
    "u_dispersion": 0.9,
    "ratio": 0.48,
    "weight_deriv": -0.5,
}

# Relative tolerance for normalised timeseries comparison.
RTOL: float = 1e-4


def load_stimulus():
    """Return the standard bar stimulus (design: 200 x 101 x 101, FOV: 20 deg)."""
    return load_2d_prf_bar_stimulus()


def make_params() -> pd.DataFrame:
    """Return a one-row DataFrame with the canonical tutorial parameters."""
    return pd.DataFrame(
        {
            "mu_x": [MU_X],
            "mu_y": [MU_Y],
            "sigma": [SIGMA],
            **{k: [v] for k, v in HRF_PARAMS.items()},
            "baseline": [BASELINE],
            "amplitude": [AMPLITUDE],
        },
    )


def prfmodel_response(stimulus, params: pd.DataFrame, *, with_hrf: bool = False) -> np.ndarray:
    """Return prfmodel's 2D Gaussian pRF prediction as a 1-D NumPy array.

    When with_hrf=False (default) the impulse and temporal models are disabled,
    returning only the normalised Gaussian RF projected onto the stimulus design.
    """
    model = Gaussian2DPRFModel() if with_hrf else Gaussian2DPRFModel(impulse_model=None, temporal_model=None)
    return np.asarray(model(stimulus, params)).ravel()


def normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize a 1-D array to unit absolute sum."""
    arr = np.asarray(arr, dtype=float).ravel()
    total = np.abs(arr).sum()
    return arr / total if total > 0 else arr


def check_and_exit(a: np.ndarray, b: np.ndarray, package: str) -> None:
    """Compare two timeseries and exit with 0 (pass) or 1 (fail).

    Both arrays are normalized to unit absolute sum before comparison
    to handle RF scale differences across packages.
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    passed = np.allclose(a_norm, b_norm, rtol=RTOL)
    max_diff = float(np.abs(a_norm - b_norm).max())
    label = f"prfmodel vs {package} (pre-HRF, normalised)"
    if passed:
        print(f"PASS  {label}  max_diff={max_diff:.2e}  rtol={RTOL}")
        sys.exit(0)
    else:
        print(f"FAIL  {label}  max_diff={max_diff:.2e}  rtol={RTOL}")
        sys.exit(1)
