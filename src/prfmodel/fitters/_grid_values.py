"""Parameter values for grid search derived from the stimulus."""

import warnings
import numpy as np
from prfmodel._docstring import doc
from prfmodel.stimuli import PRFStimulus

_MAX_ALIGNMENT_SEARCH = 1000


class GridAlignmentWarning(UserWarning):
    """Warning for when the edges or the midpoint of a stimulus do not fall on grid values."""


def _is_aligned(values: np.ndarray, targets: tuple[float, ...]) -> bool:
    """Check whether all targets coincide with one of the values (up to floating point error)."""
    tolerance = 1e-6 * (values[-1] - values[0]) / max(len(values) - 1, 1)
    return all(np.any(np.isclose(values, target, rtol=0.0, atol=tolerance)) for target in targets)


def _mu_values(coordinates: np.ndarray, num_mu: int, mu_extent: float, name: str) -> np.ndarray:
    """Compute grid values for a pRF center coordinate and warn if they are not aligned with the stimulus."""
    if num_mu < 1:
        msg = f"'num_mu' must be at least 1, but is {num_mu}"
        raise ValueError(msg)

    if mu_extent <= 0.0:
        msg = f"'mu_extent' must be positive, but is {mu_extent}"
        raise ValueError(msg)

    stim_min, stim_max = float(np.min(coordinates)), float(np.max(coordinates))
    midpoint = (stim_min + stim_max) / 2.0
    half_range = mu_extent * (stim_max - stim_min) / 2.0

    def make_values(num: int) -> np.ndarray:
        return np.linspace(midpoint - half_range, midpoint + half_range, num)

    values = make_values(num_mu)
    targets = (stim_min, midpoint, stim_max)

    if not _is_aligned(values, targets):
        candidates = range(num_mu + 1, num_mu + _MAX_ALIGNMENT_SEARCH)
        suggestion = next((num for num in candidates if _is_aligned(make_values(num), targets)), None)
        msg = f"The edges or the midpoint of the stimulus do not fall on the grid values of '{name}'"
        if suggestion is not None:
            msg += f"; use 'num_mu={suggestion}' to align them"
        warnings.warn(msg, category=GridAlignmentWarning, stacklevel=3)

    return values


def _sigma_values(
    coordinates: list[np.ndarray],
    num_sigma: int,
    sigma_range: tuple[float, float] | None,
    log_sigma: bool,
) -> np.ndarray:
    """Compute grid values for the pRF size."""
    if sigma_range is None:
        # From the resolution of the stimulus to its full extent
        steps = [np.diff(np.unique(coords)) for coords in coordinates]
        sigma_min = float(min(np.min(step) for step in steps if step.size > 0))
        sigma_max = float(max(np.ptp(coords) for coords in coordinates))
    else:
        sigma_min, sigma_max = sigma_range

    if log_sigma:
        if sigma_min <= 0.0:
            msg = f"The lower limit of 'sigma_range' must be positive when 'log_sigma=True', but is {sigma_min}"
            raise ValueError(msg)
        return np.geomspace(sigma_min, sigma_max, num_sigma)

    return np.linspace(sigma_min, sigma_max, num_sigma)


def _check_dimensions(stimulus: PRFStimulus, expected: int) -> None:
    num_dim = stimulus.grid.shape[-1]

    if num_dim != expected:
        msg = f"Stimulus must be {expected}-dimensional, but has {num_dim} dimensions"
        raise ValueError(msg)


@doc
def grid_values_2d_prf(  # noqa: PLR0913
    stimulus: PRFStimulus,
    num_mu: int = 21,
    mu_extent: float = 2.0,
    num_sigma: int = 20,
    sigma_range: tuple[float, float] | None = None,
    log_sigma: bool = True,
) -> dict[str, np.ndarray]:
    """Create grid search values for the center and size of a two-dimensional population receptive field (pRF).

    The values for the pRF center coordinates `mu_x` and `mu_y` span the stimulus, extended beyond it by `mu_extent`
    around its midpoint. The values for the pRF size `sigma` range from the resolution to the extent of the stimulus
    by default.

    Parameters
    ----------
    %(stimulus_prf)s
    num_mu : int, optional
        Number of values for each of `mu_x` and `mu_y`.
    mu_extent : float, optional
        Range of the pRF center values relative to the range of the stimulus. With `1.0`, the values span exactly the
        stimulus; with `2.0` (the default), they span twice its range around its midpoint.
    num_sigma : int, optional
        Number of values for `sigma`.
    sigma_range : tuple[float, float] or None, optional
        Smallest and largest value for `sigma`. If `None`, the smallest value is the spacing of the stimulus grid and
        the largest value is the largest extent of the stimulus (in `x` or `y`).
    log_sigma : bool, optional
        Whether the values for `sigma` are log-spaced (the default) or linearly spaced.

    Returns
    -------
    dict[str, numpy.ndarray]
        Values for `mu_x`, `mu_y`, and `sigma` that can be passed to :meth:`prfmodel.fitters.GridFitter.fit`.
        Values for the other model parameters (e.g., `baseline` and `amplitude`) must be added, for example, with
        the `|` operator.

    Raises
    ------
    ValueError
        If the stimulus is not 2-dimensional, `num_mu` is smaller than one, `mu_extent` is not positive, or the lower
        limit of `sigma_range` is not positive when `log_sigma=True`.

    Warns
    -----
    GridAlignmentWarning
        If the edges or the midpoint of the stimulus do not fall on the values for `mu_x` or `mu_y`. Aligned values
        make it possible to tell exactly whether an estimated pRF center lies inside the stimulus. The warning
        suggests a value for `num_mu` that aligns them.

    See Also
    --------
    grid_values_1d_prf : Grid search values for one-dimensional pRFs.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> stimulus = load_2d_prf_bar_stimulus()  # Spans about -4 to 4 degrees in x and y
    >>> values = grid_values_2d_prf(stimulus, num_mu=9, num_sigma=5)
    >>> np.round(values["mu_x"], 2)
    array([-8.01, -6.01, -4.01, -2.  ,  0.  ,  2.  ,  4.01,  6.01,  8.01])
    >>> np.round(values["sigma"], 2)
    array([0.06, 0.21, 0.71, 2.39, 8.01])
    >>> parameter_values = values | {"baseline": [0.0], "amplitude": [1.0]}

    """
    _check_dimensions(stimulus, 2)

    # The grid stores y in grid[..., 0] (varying along rows) and x in grid[..., 1] (varying along columns)
    y_coordinates = stimulus.grid[:, 0, 0]
    x_coordinates = stimulus.grid[0, :, 1]

    return {
        "mu_x": _mu_values(x_coordinates, num_mu, mu_extent, "mu_x"),
        "mu_y": _mu_values(y_coordinates, num_mu, mu_extent, "mu_y"),
        "sigma": _sigma_values([x_coordinates, y_coordinates], num_sigma, sigma_range, log_sigma),
    }


def grid_values_1d_prf(  # noqa: PLR0913
    stimulus: PRFStimulus,
    num_mu: int = 49,
    mu_extent: float = 2.0,
    num_sigma: int = 50,
    sigma_range: tuple[float, float] | None = None,
    log_sigma: bool = True,
) -> dict[str, np.ndarray]:
    """Create grid search values for the center and size of a one-dimensional pRF from the stimulus.

    The values for the pRF center `mu` span the stimulus coordinates, extended beyond them by `mu_extent` around their
    midpoint. The values for the pRF size `sigma` range from the smallest spacing to the full range of the stimulus
    coordinates by default.

    Parameters
    ----------
    stimulus : PRFStimulus
        One-dimensional stimulus. The coordinates do not need to be regularly spaced.
    num_mu : int, optional
        Number of values for `mu`.
    mu_extent : float, optional
        Range of the pRF center values relative to the range of the stimulus coordinates. With `1.0`, the values span
        exactly the stimulus; with `2.0` (the default), they span twice its range around its midpoint.
    num_sigma : int, optional
        Number of values for `sigma`.
    sigma_range : tuple[float, float] or None, optional
        Smallest and largest value for `sigma`. If `None`, the smallest value is the smallest spacing between the
        stimulus coordinates and the largest value is their range.
    log_sigma : bool, optional
        Whether the values for `sigma` are log-spaced (the default) or linearly spaced. Log spacing samples small pRFs,
        whose predictions change quickly with their size, more densely.

    Returns
    -------
    dict[str, numpy.ndarray]
        Values for `mu` and `sigma` that can be passed to :meth:`prfmodel.fitters.GridFitter.fit`. Values for the
        other model parameters (e.g., `baseline` and `amplitude`) must be added, for example, with the `|` operator.

    Raises
    ------
    ValueError
        If the stimulus is not 1-dimensional, `num_mu` is smaller than one, `mu_extent` is not positive, or the lower
        limit of `sigma_range` is not positive when `log_sigma=True`.

    Warns
    -----
    GridAlignmentWarning
        If the edges or the midpoint of the stimulus do not fall on the values for `mu`. The warning suggests a value
        for `num_mu` that aligns them.

    See Also
    --------
    grid_values_2d_prf : Grid search values for two-dimensional pRFs.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.examples import load_1d_prf_lognumerosity_stimulus
    >>> stimulus = load_1d_prf_lognumerosity_stimulus()  # Log numerosities from log(1) to log(20)
    >>> values = grid_values_1d_prf(stimulus, num_mu=9, num_sigma=5)
    >>> np.round(values["mu"], 2)
    array([-1.5 , -0.75,  0.  ,  0.75,  1.5 ,  2.25,  3.  ,  3.74,  4.49])
    >>> np.round(values["sigma"], 2)
    array([0.15, 0.32, 0.68, 1.43, 3.  ])

    """
    _check_dimensions(stimulus, 1)

    coordinates = stimulus.grid[..., 0]

    return {
        "mu": _mu_values(coordinates, num_mu, mu_extent, "mu"),
        "sigma": _sigma_values([coordinates], num_sigma, sigma_range, log_sigma),
    }
