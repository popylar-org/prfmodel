"""Parameter values for grid search derived from the stimulus."""

import warnings
import numpy as np
from prfmodel._docstring import doc
from prfmodel.stimuli import PRFStimulus

_MAX_ALIGNMENT_SEARCH = 1000

# Default radius of the smallest ring of a polar grid, in stimulus grid spacings. It does not depend on the number of
# rings, so that more rings always make the grid denser, including near the origin.
_SMALLEST_ECC_SPACINGS = 2.0


class GridAlignmentWarning(UserWarning):
    """Warning for when the edges or the midpoint of a stimulus do not fall on grid values.

    For a polar grid, the warning is raised when the midpoint of the stimulus is not at (0, 0), the center of the rings,
    or when a given number of angles does not include all meridians.
    """


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


def _resolution(coordinates: list[np.ndarray]) -> float:
    """Return the smallest spacing between the stimulus coordinates across all dimensions."""
    steps = [np.diff(np.unique(coords)) for coords in coordinates]
    return float(min(np.min(step) for step in steps if step.size > 0))


def _extent(coordinates: list[np.ndarray]) -> float:
    """Return the largest range of the stimulus coordinates across all dimensions."""
    return float(max(np.ptp(coords) for coords in coordinates))


def _sigma_values(
    coordinates: list[np.ndarray],
    num_sigma: int,
    sigma_range: tuple[float, float] | None,
    log_sigma: bool,
) -> np.ndarray:
    """Compute grid values for the pRF size."""
    if sigma_range is None:
        # From the resolution of the stimulus to its full extent
        sigma_min = _resolution(coordinates)
        sigma_max = _extent(coordinates)
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
    grid_values_2d_prf_polar : Polar grid search values for two-dimensional pRFs.
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


@doc
def grid_values_2d_prf_polar(  # noqa: PLR0913
    stimulus: PRFStimulus,
    num_ecc: int = 16,
    mu_extent: float = 2.0,
    ecc_range: tuple[float, float] | None = None,
    num_angle: int | None = None,
    num_sigma: int = 20,
    sigma_range: tuple[float, float] | None = None,
    log_sigma: bool = True,
) -> dict[str, np.ndarray]:
    r"""Create polar grid search values for the center and size of a two-dimensional population receptive field (pRF).

    Instead of a rectangular grid of `mu_x` and `mu_y` values, the pRF centers are placed on rings around the point
    (0, 0) of the stimulus coordinates, which must be the midpoint of the stimulus (normally the fixation point).
    The values for the eccentricity `ecc` are log-spaced, so that the rings are dense near the origin,
    where pRFs tend to be small, and sparse in the periphery, where they tend to be large.
    The values for the polar angle `angle` cover the full circle. Every combination of an `ecc` and an `angle` value is
    one pRF center. The values for the pRF size `sigma` are the same as those of :func:`grid_values_2d_prf`.

    Parameters
    ----------
    %(stimulus_prf)s
    num_ecc : int, optional
        Number of values (rings) for `ecc`. Must be at least two.
    mu_extent : float, optional
        Largest eccentricity relative to half the largest extent of the stimulus (in `x` or `y`). With `1.0`, the
        outer ring reaches the edges of the stimulus along its wider axis; with `2.0` (the default), it reaches as far
        as the default center values of :func:`grid_values_2d_prf` along that axis. Ignored if `ecc_range` is given.
    ecc_range : tuple[float, float] or None, optional
        Smallest and largest value for `ecc`. If `None`, the largest value is set by `mu_extent` and the smallest
        value is two stimulus grid spacings (see Notes).
    num_angle : int or None, optional
        Number of values for `angle`. If `None`, it is chosen so that the grid cells are roughly square, that is, the
        distance between neighboring centers on a ring is about the same as the distance between neighboring rings.
        It is rounded to a multiple of four, so that there are centers on the horizontal and vertical meridians.
        A given value is used as is, with a warning if it is not a multiple of four.
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
        Values for `ecc`, `angle`, and `sigma` that can be passed to :meth:`prfmodel.fitters.GridFitter.fit` for a
        model whose pRF center is parameterized by eccentricity and polar angle. The angle is in radians and runs
        counterclockwise from the positive x-axis, in the range :math:`[-\pi, \pi)`, like
        :func:`prfmodel.utils.calculate_polar_angle`. The center coordinates are
        `mu_x = ecc * cos(angle)` and `mu_y = ecc * sin(angle)`. Values for the other model parameters (e.g.,
        `baseline` and `amplitude`) must be added, for example, with the `|` operator.

    Raises
    ------
    ValueError
        If the stimulus is not 2-dimensional, `num_ecc` is smaller than two, `mu_extent` is not positive, `num_angle`
        is smaller than one, the smallest eccentricity is not positive or not smaller than the largest, or the lower
        limit of `sigma_range` is not positive when `log_sigma=True`.

    Warns
    -----
    GridAlignmentWarning
        If the midpoint of the stimulus is not at (0, 0) (by more than half the spacing of the stimulus grid),
        because the rings are centered on (0, 0). Also if a given `num_angle` is not a multiple of four, so that not
        all meridians fall on grid values. The warning suggests a value for `num_angle` that aligns them.

    See Also
    --------
    grid_values_2d_prf : Rectangular grid search values for two-dimensional pRFs.
    prfmodel.utils.calculate_eccentricity : Eccentricity of two-dimensional pRF centers.
    prfmodel.utils.calculate_polar_angle : Polar angle of two-dimensional pRF centers.

    Notes
    -----
    With log-spaced rings, the distance between two neighboring rings at eccentricity :math:`r` is about
    :math:`r \ln q`, where :math:`q` is the ratio between neighboring eccentricities. The distance between
    neighboring centers on a ring is :math:`2 \pi r / n`, where :math:`n` is the number of angles. Both grow in
    proportion to :math:`r`, so the grid cells keep their shape across eccentricities, and they are square for
    :math:`n = 2 \pi / \ln q`. The default for `num_angle` is this value rounded to a multiple of four.

    By default, the smallest ring has a radius of two stimulus grid spacings, whatever the number of rings. pRFs
    closer to the origin than the smallest ring still have candidates on it, at most two grid spacings away. Because
    the smallest ring does not move with `num_ecc`, more rings always make the grid denser, also near the origin.
    With many rings, neighboring centers near the origin can be closer together than the stimulus can resolve, so
    some of their predictions are nearly identical; this costs candidates but not accuracy.
    The outer edge of the grid is a circle, so with `mu_extent=1.0` the corners of a rectangular stimulus are not
    covered.

    Examples
    --------
    >>> import numpy as np
    >>> from prfmodel.examples import load_2d_prf_bar_stimulus
    >>> stimulus = load_2d_prf_bar_stimulus()  # Spans about -4 to 4 degrees in x and y
    >>> values = grid_values_2d_prf_polar(stimulus, num_ecc=5, mu_extent=1.0, num_sigma=5)
    >>> np.round(values["ecc"], 1)
    array([0.1, 0.3, 0.7, 1.7, 4. ])
    >>> len(values["angle"])  # Chosen for roughly square grid cells
    8
    >>> parameter_values = values | {"baseline": [0.0], "amplitude": [1.0]}

    """
    _check_dimensions(stimulus, 2)

    # Log spacing needs at least two rings to define the ratio between them
    if num_ecc < 2:  # noqa: PLR2004
        msg = f"'num_ecc' must be at least 2, but is {num_ecc}"
        raise ValueError(msg)

    if mu_extent <= 0.0:
        msg = f"'mu_extent' must be positive, but is {mu_extent}"
        raise ValueError(msg)

    if num_angle is not None and num_angle < 1:
        msg = f"'num_angle' must be at least 1, but is {num_angle}"
        raise ValueError(msg)

    # The grid stores y in grid[..., 0] (varying along rows) and x in grid[..., 1] (varying along columns)
    y_coordinates = stimulus.grid[:, 0, 0]
    x_coordinates = stimulus.grid[0, :, 1]
    coordinates = [x_coordinates, y_coordinates]

    resolution = _resolution(coordinates)
    half_extent = _extent(coordinates) / 2.0

    # The rings are centered on the point (0, 0) of the stimulus coordinates, because the polar model measures
    # eccentricity and angle from there. The midpoint of the stimulus (normally the fixation point) must be at (0, 0).
    midpoint_x, midpoint_y = ((float(np.min(coords)) + float(np.max(coords))) / 2.0 for coords in coordinates)
    if np.hypot(midpoint_x, midpoint_y) > resolution / 2.0:
        msg = (
            f"The midpoint of the stimulus (x={midpoint_x:.3g}, y={midpoint_y:.3g}) is not at (0, 0), "
            "where the rings of 'ecc' are centered"
        )
        warnings.warn(msg, category=GridAlignmentWarning, stacklevel=2)

    if ecc_range is None:
        ecc_max = mu_extent * half_extent
        ecc_min = _SMALLEST_ECC_SPACINGS * resolution
    else:
        ecc_min, ecc_max = ecc_range

    if not 0.0 < ecc_min < ecc_max:
        msg = (
            "The smallest eccentricity must be positive and smaller than the largest, "
            f"but they are {ecc_min} and {ecc_max}"
        )
        raise ValueError(msg)

    ecc = np.geomspace(ecc_min, ecc_max, num_ecc)

    if num_angle is None:
        # Square grid cells: the spacing on a ring (2 pi r / n) equals the spacing between rings (r ln q).
        # Rounded to a multiple of four, so that the angles include 0, pi/2, pi, and -pi/2 (the meridians)
        num_angle = 4 * max(1, int(np.round(2.0 * np.pi / np.log(ecc[1] / ecc[0]) / 4.0)))
    elif num_angle % 4 != 0:
        suggestion = 4 * max(1, round(num_angle / 4))
        msg = f"Not all meridians fall on the grid values of 'angle'; use 'num_angle={suggestion}' to align them"
        if num_angle % 2 != 0:
            msg += " (an odd number also makes the grid asymmetric left and right)"
        warnings.warn(msg, category=GridAlignmentWarning, stacklevel=2)

    angle = np.linspace(-np.pi, np.pi, num_angle, endpoint=False)

    return {
        "ecc": ecc,
        "angle": angle,
        "sigma": _sigma_values(coordinates, num_sigma, sigma_range, log_sigma),
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
