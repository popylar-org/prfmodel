"""Validate prfmodel's 2D Gaussian pRF against braincoder.

Comparison strategy
-------------------
We compare the pre-HRF neural response: the dot product of the Gaussian RF
with the stimulus design matrix.

- prfmodel: Gaussian2DPRFModel(impulse_model=None, temporal_model=None)
- braincoder: GaussianPRF2D.predict() - this class has no HRF convolution,
  so the prediction is already the RF-weighted stimulus projection.

Both timeseries are z-scored before Pearson r is computed;
we assert r > MIN_PEARSON_R.

Coordinate conventions
----------------------
prfmodel stimulus grid: shape (H, W, 2), last axis = [y, x] in visual-angle degrees.
braincoder GaussianPRF2D:
  - paradigm: (n_timepoints, H, W) - same layout as prfmodel's design
  - grid_coordinates: DataFrame with columns ['x', 'y'] in visual-angle degrees,
    one row per spatial pixel (H*W rows total, row-major / C order)
  - parameters: DataFrame with columns ['x', 'y', 'sd', 'amplitude', 'baseline']
    where 'x'/'y' are the RF centre and 'sd' is sigma

Installation
------------
pip install git+https://github.com/Gilles86/braincoder.git
(requires tensorflow and tensorflow-probability)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from braincoder.models import GaussianPRF2D
except ImportError:
    print(
        "ERROR: braincoder is not installed.\n"
        "Install with: pip install git+https://github.com/Gilles86/braincoder.git\n"
        "Note: braincoder requires tensorflow and tensorflow-probability.",
    )
    sys.exit(1)

# Force float32 globally so all TF operations use a consistent dtype.
tf.keras.backend.set_floatx("float32")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared import AMPLITUDE
from shared import BASELINE
from shared import MU_X
from shared import MU_Y
from shared import SIGMA
from shared import check_and_exit
from shared import load_stimulus
from shared import make_params
from shared import prfmodel_response


def _braincoder_response(stimulus) -> np.ndarray:
    """Compute pre-HRF response using braincoder's GaussianPRF2D.

    Returns a 1-D array of length n_frames.
    """
    # prfmodel grid: (H, W, 2) with [:,:,0]=y, [:,:,1]=x.
    # braincoder expects rows=pixels, columns=['x', 'y'].
    grid_flat = stimulus.grid.reshape(-1, 2)  # (H*W, 2)
    grid_coordinates = pd.DataFrame(
        {"x": grid_flat[:, 1], "y": grid_flat[:, 0]},  # swap [y, x] -> [x, y]
    ).astype("float32")

    bc_params = pd.DataFrame(
        {
            "x": [MU_X],
            "y": [MU_Y],
            "sd": [SIGMA],
            "amplitude": [AMPLITUDE],
            "baseline": [BASELINE],
        },
    ).astype("float32")

    model = GaussianPRF2D(
        paradigm=stimulus.design.astype("float32"),
        grid_coordinates=grid_coordinates,
    )
    prediction = model.predict(parameters=bc_params)  # DataFrame (T, 1)

    return np.asarray(prediction).ravel()


def main() -> None:
    """Run braincoder comparison and exit with 0 on pass, 1 on fail."""
    stimulus = load_stimulus()
    params = make_params()

    ref = prfmodel_response(stimulus, params, with_hrf=False)
    bc = _braincoder_response(stimulus)

    check_and_exit(ref, bc, "braincoder")


if __name__ == "__main__":
    main()
