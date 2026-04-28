"""Validate prfmodel's 2D Gaussian pRF against prfpy.

Comparison strategy
-------------------
We compare the pre-HRF neural response: the dot product of the Gaussian RF
with the stimulus design matrix, summed over the spatial dimensions.

- prfmodel: Gaussian2DPRFModel(impulse_model=None, temporal_model=None)
  normalised RF: exp(-d^2/2*sigma^2) / (2*pi*sigma^2)
- prfpy:    gauss2D_iso_cart -> unnormalised RF: exp(-d^2/2*sigma^2)

Because the overall scale differs (prfmodel divides by the Gaussian volume),
we z-score both timeseries before computing Pearson r. A true implementation
match -> r approx 1; we assert r > shared.MIN_PEARSON_R (default 0.9999).

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
from shared import MU_X
from shared import MU_Y
from shared import SIGMA
from shared import check_and_exit
from shared import load_stimulus
from shared import make_params
from shared import pearson_r
from shared import prfmodel_response


def _prfpy_response(stimulus) -> np.ndarray:
    """Compute pre-HRF response using prfpy's gauss2D_iso_cart.

    Returns a 1-D array of length n_frames (unnormalised scale).
    """
    # Extract x and y coordinate grids from the prfmodel stimulus.
    # grid[:, :, 0] = y-axis, grid[:, :, 1] = x-axis (dimension_labels=['y', 'x'])
    y_grid = stimulus.grid[:, :, 0]  # shape (H, W)
    x_grid = stimulus.grid[:, :, 1]  # shape (H, W)

    # gauss2D_iso_cart returns exp(-((x-mu_x)^2 + (y-mu_y)^2) / (2*sigma^2)) - unnormalised.
    rf = gauss2D_iso_cart(x=x_grid, y=y_grid, mu=(MU_X, MU_Y), sigma=SIGMA)  # (H, W)

    # Project RF onto stimulus: sum over spatial dims for each time frame.
    # design shape: (T, H, W)
    return np.einsum("thw,hw->t", stimulus.design, rf)  # (T,)


def main() -> None:
    """Run prfpy comparison and exit with 0 on pass, 1 on fail."""
    stimulus = load_stimulus()
    params = make_params()

    ref = prfmodel_response(stimulus, params, with_hrf=False)
    prfpy = _prfpy_response(stimulus)

    r = pearson_r(ref, prfpy)
    check_and_exit(r, "prfpy")


if __name__ == "__main__":
    main()
