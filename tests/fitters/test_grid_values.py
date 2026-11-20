"""Tests for grid search values derived from the stimulus."""

import warnings
import numpy as np
import pytest
from prfmodel.fitters import GridAlignmentWarning
from prfmodel.fitters import grid_values_1d_prf
from prfmodel.fitters import grid_values_2d_prf
from prfmodel.fitters import grid_values_2d_prf_polar
from prfmodel.stimuli import PRFStimulus


def _stimulus_2d(x: np.ndarray, y: np.ndarray) -> PRFStimulus:
    xv, yv = np.meshgrid(x, y)
    # y is stacked first because it varies along the row axis
    grid = np.stack((yv, xv), axis=-1)
    return PRFStimulus(design=np.zeros((3, len(y), len(x))), grid=grid, frame_duration=1.0)


def _stimulus_1d(coordinates: np.ndarray) -> PRFStimulus:
    return PRFStimulus(design=np.eye(len(coordinates)), grid=np.asarray(coordinates)[:, np.newaxis], frame_duration=1.0)


@pytest.fixture
def stimulus_2d() -> PRFStimulus:
    """Non-square, asymmetric 2D stimulus with x in [0, 8] and y in [-1, 1]."""
    return _stimulus_2d(np.linspace(0.0, 8.0, 17), np.linspace(-1.0, 1.0, 5))


def test_grid_values_2d_prf_keys(stimulus_2d: PRFStimulus):
    """Test that values are returned for the pRF center and size."""
    values = grid_values_2d_prf(stimulus_2d)
    assert list(values) == ["mu_x", "mu_y", "sigma"]
    assert len(values["mu_x"]) == len(values["mu_y"]) == 21  # noqa: PLR2004
    assert len(values["sigma"]) == 20  # noqa: PLR2004


@pytest.mark.parametrize(("mu_extent", "num_mu"), [(1.0, 5), (2.0, 9), (1.5, 7)])
def test_grid_values_2d_prf_extent(stimulus_2d: PRFStimulus, mu_extent: float, num_mu: int):
    """Test that the center values span the stimulus range scaled around its midpoint for each axis separately."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GridAlignmentWarning)
        values = grid_values_2d_prf(stimulus_2d, num_mu=num_mu, mu_extent=mu_extent)
    np.testing.assert_allclose(values["mu_x"][[0, -1]], [4.0 - 4.0 * mu_extent, 4.0 + 4.0 * mu_extent])
    np.testing.assert_allclose(values["mu_y"][[0, -1]], [-mu_extent, mu_extent])
    assert len(values["mu_x"]) == num_mu


def test_grid_values_2d_prf_aligned(stimulus_2d: PRFStimulus):
    """Test that the stimulus edges and midpoint fall on aligned grid values without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridAlignmentWarning)
        values = grid_values_2d_prf(stimulus_2d, num_mu=21, mu_extent=2.0)
    for target in (0.0, 4.0, 8.0):
        assert np.any(np.isclose(values["mu_x"], target))
    for target in (-1.0, 0.0, 1.0):
        assert np.any(np.isclose(values["mu_y"], target))


def test_grid_values_2d_prf_reproduces_manual_grid():
    """Test that the defaults reproduce a grid extended to twice the stimulus range with 21 values."""
    stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=5, width=64, height=64, frame_duration=1.0)
    grid_min, grid_max = stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))
    values = grid_values_2d_prf(stimulus)
    np.testing.assert_allclose(values["mu_x"], np.linspace(2 * grid_min[1], 2 * grid_max[1], 21))
    np.testing.assert_allclose(values["mu_y"], np.linspace(2 * grid_min[0], 2 * grid_max[0], 21))


@pytest.mark.parametrize(("num_mu", "suggestion"), [(20, 21), (8, 9)])
def test_grid_values_2d_prf_misaligned_warning(stimulus_2d: PRFStimulus, num_mu: int, suggestion: int):
    """Test that misaligned center values raise a warning that suggests an aligned number of values."""
    with pytest.warns(GridAlignmentWarning, match=f"num_mu={suggestion}"):
        grid_values_2d_prf(stimulus_2d, num_mu=num_mu, mu_extent=2.0)


def test_grid_values_2d_prf_default_sigma(stimulus_2d: PRFStimulus):
    """Test that the default sigma values range log-spaced from the grid spacing to the largest stimulus extent."""
    sigma = grid_values_2d_prf(stimulus_2d, num_sigma=5)["sigma"]
    np.testing.assert_allclose(sigma[[0, -1]], [0.5, 8.0])
    np.testing.assert_allclose(np.diff(np.log(sigma)), np.log(16.0) / 4)


def test_grid_values_2d_prf_linear_sigma(stimulus_2d: PRFStimulus):
    """Test that sigma values can be linearly spaced with a custom range."""
    sigma = grid_values_2d_prf(stimulus_2d, num_sigma=4, sigma_range=(0.0, 3.0), log_sigma=False)["sigma"]
    np.testing.assert_allclose(sigma, [0.0, 1.0, 2.0, 3.0])


def test_grid_values_2d_prf_invalid(stimulus_2d: PRFStimulus):
    """Test that invalid arguments raise errors."""
    with pytest.raises(ValueError, match="2-dimensional"):
        grid_values_2d_prf(_stimulus_1d(np.arange(3.0)))
    with pytest.raises(ValueError, match="num_mu"):
        grid_values_2d_prf(stimulus_2d, num_mu=0)
    with pytest.raises(ValueError, match="mu_extent"):
        grid_values_2d_prf(stimulus_2d, mu_extent=0.0)
    with pytest.raises(ValueError, match="positive"):
        grid_values_2d_prf(stimulus_2d, sigma_range=(0.0, 1.0))


def test_grid_values_1d_prf():
    """Test that 1D values span irregular stimulus coordinates and sigma ranges from their smallest spacing."""
    stimulus = _stimulus_1d(np.log(np.array([1.0, 2.0, 3.0, 20.0])))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridAlignmentWarning)
        values = grid_values_1d_prf(stimulus)
    assert list(values) == ["mu", "sigma"]
    # Twice the stimulus range around its midpoint
    np.testing.assert_allclose(values["mu"][[0, -1]], [-np.log(20.0) / 2, 3 * np.log(20.0) / 2])
    assert len(values["mu"]) == 49  # noqa: PLR2004
    np.testing.assert_allclose(values["sigma"][[0, -1]], [np.log(1.5), np.log(20.0)])
    # Log-spaced
    np.testing.assert_allclose(np.diff(np.log(values["sigma"])), np.diff(np.log(values["sigma"]))[0])


def test_grid_values_1d_prf_extent():
    """Test that the 1D center values can span exactly the stimulus with linearly spaced sigma values."""
    values = grid_values_1d_prf(_stimulus_1d(np.arange(5.0)), num_mu=5, mu_extent=1.0, num_sigma=4, log_sigma=False)
    np.testing.assert_allclose(values["mu"], [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(values["sigma"], [1.0, 2.0, 3.0, 4.0])


def test_grid_values_1d_prf_misaligned_warning():
    """Test that misaligned 1D center values raise a warning."""
    with pytest.warns(GridAlignmentWarning, match="'mu'"):
        grid_values_1d_prf(_stimulus_1d(np.arange(5.0)), num_mu=50)


def test_grid_values_1d_prf_invalid():
    """Test that a stimulus that is not 1-dimensional raises an error."""
    with pytest.raises(ValueError, match="1-dimensional"):
        grid_values_1d_prf(_stimulus_2d(np.arange(3.0), np.arange(3.0)))

@pytest.fixture
def stimulus_centered() -> PRFStimulus:
    """Non-square 2D stimulus centered on (0, 0) with x in [-4, 4], y in [-2, 2], and a grid spacing of 0.5."""
    return _stimulus_2d(np.linspace(-4.0, 4.0, 17), np.linspace(-2.0, 2.0, 9))


def test_grid_values_2d_prf_polar_keys(stimulus_centered: PRFStimulus):
    """Test that values are returned for the eccentricity, polar angle, and size."""
    values = grid_values_2d_prf_polar(stimulus_centered)
    assert list(values) == ["ecc", "angle", "sigma"]
    assert len(values["ecc"]) == 16  # noqa: PLR2004
    assert len(values["sigma"]) == 20  # noqa: PLR2004


def test_grid_values_2d_prf_polar_default_ecc(stimulus_centered: PRFStimulus):
    """Test that neighboring centers on the smallest ring are one grid spacing apart and the rings are log-spaced."""
    ecc = grid_values_2d_prf_polar(stimulus_centered, num_ecc=5)["ecc"]
    ratio = ecc[1] / ecc[0]

    # Spacing on the smallest ring with square cells equals the grid spacing (0.5)
    np.testing.assert_allclose(ecc[0] * np.log(ratio), 0.5)
    np.testing.assert_allclose(ecc[-1], 8.0)
    np.testing.assert_allclose(ecc[1:] / ecc[:-1], ratio)


@pytest.mark.parametrize("mu_extent", [1.0, 1.5])
def test_grid_values_2d_prf_polar_mu_extent(stimulus_centered: PRFStimulus, mu_extent: float):
    """Test that the largest ring is half the largest stimulus extent scaled by mu_extent."""
    ecc = grid_values_2d_prf_polar(stimulus_centered, mu_extent=mu_extent)["ecc"]
    np.testing.assert_allclose(ecc[-1], 4.0 * mu_extent)


def test_grid_values_2d_prf_polar_ecc_range(stimulus_centered: PRFStimulus):
    """Test that ecc_range overrides the default range and mu_extent."""
    ecc = grid_values_2d_prf_polar(stimulus_centered, num_ecc=4, mu_extent=1.0, ecc_range=(0.25, 2.0))["ecc"]
    np.testing.assert_allclose(ecc, [0.25, 0.5, 1.0, 2.0])


@pytest.mark.parametrize("num_ecc", [2, 5, 19, 40])
def test_grid_values_2d_prf_polar_default_num_angle(stimulus_centered: PRFStimulus, num_ecc: int):
    """Test that the default number of angles is a multiple of four close to the one for square grid cells."""
    values = grid_values_2d_prf_polar(stimulus_centered, num_ecc=num_ecc)
    num_angle = len(values["angle"])
    square_cells = 2 * np.pi / np.log(values["ecc"][1] / values["ecc"][0])
    assert num_angle % 4 == 0
    assert abs(num_angle - square_cells) <= 2  # noqa: PLR2004


def test_grid_values_2d_prf_polar_angle_spacing(stimulus_centered: PRFStimulus):
    """Test that the angles are evenly spaced over the full circle from -pi, without the duplicate at pi."""
    angle = grid_values_2d_prf_polar(stimulus_centered, num_angle=8)["angle"]
    np.testing.assert_allclose(angle, np.pi * np.arange(-4, 4) / 4, atol=1e-12)


def test_grid_values_2d_prf_polar_explicit_num_angle(stimulus_centered: PRFStimulus):
    """Test that a given number of angles is used as is, with a warning if it is not a multiple of four."""
    with pytest.warns(GridAlignmentWarning, match="num_angle=8"):
        angle = grid_values_2d_prf_polar(stimulus_centered, num_angle=7)["angle"]
    assert len(angle) == 7  # noqa: PLR2004


def test_grid_values_2d_prf_polar_aligned_num_angle_no_warning(stimulus_centered: PRFStimulus):
    """Test that a given multiple of four does not raise a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridAlignmentWarning)
        grid_values_2d_prf_polar(stimulus_centered, num_angle=12)


def test_grid_values_2d_prf_polar_symmetric(stimulus_centered: PRFStimulus):
    """Test that the default grid has centers on the meridians and is mirror symmetric in x and y."""
    values = grid_values_2d_prf_polar(stimulus_centered)

    for direction in (-np.pi, -np.pi / 2, 0.0, np.pi / 2):
        assert np.any(np.isclose(values["angle"], direction, atol=1e-12))

    ecc, angle = np.meshgrid(values["ecc"], values["angle"], indexing="ij")
    mu_x = (ecc * np.cos(angle)).ravel()
    mu_y = (ecc * np.sin(angle)).ravel()
    points = np.stack([mu_x, mu_y], axis=1)

    for mirrored in (np.stack([-mu_x, mu_y], axis=1), np.stack([mu_x, -mu_y], axis=1)):
        # Distance from every mirrored point to the closest original point
        dist = np.linalg.norm(mirrored[:, None, :] - points[None, :, :], axis=-1)
        np.testing.assert_allclose(dist.min(axis=1), 0.0, atol=1e-9)


@pytest.mark.parametrize("kwargs", [{}, {"num_sigma": 7, "sigma_range": (0.1, 3.0), "log_sigma": False}])
def test_grid_values_2d_prf_polar_sigma(stimulus_centered: PRFStimulus, kwargs: dict):
    """Test that the sigma values are the same as those of grid_values_2d_prf."""
    np.testing.assert_allclose(
        grid_values_2d_prf_polar(stimulus_centered, **kwargs)["sigma"],
        grid_values_2d_prf(stimulus_centered, **kwargs)["sigma"],
    )


def test_grid_values_2d_prf_polar_off_center_warning(stimulus_2d: PRFStimulus):
    """Test that a stimulus whose midpoint is not at (0, 0) raises a warning."""
    with pytest.warns(GridAlignmentWarning, match=r"not at \(0, 0\)"):
        grid_values_2d_prf_polar(stimulus_2d)


@pytest.mark.parametrize("offset", [0.0, 0.2])
def test_grid_values_2d_prf_polar_centered_no_warning(offset: float):
    """Test that a midpoint within half a grid spacing (0.25) of (0, 0) does not raise a warning."""
    stimulus = _stimulus_2d(np.linspace(-4.0, 4.0, 17) + offset, np.linspace(-2.0, 2.0, 9))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GridAlignmentWarning)
        grid_values_2d_prf_polar(stimulus)


def test_grid_values_2d_prf_polar_invalid(stimulus_centered: PRFStimulus):
    """Test that invalid arguments raise errors."""
    with pytest.raises(ValueError, match="2-dimensional"):
        grid_values_2d_prf_polar(_stimulus_1d(np.arange(3.0)))
    with pytest.raises(ValueError, match="num_ecc"):
        grid_values_2d_prf_polar(stimulus_centered, num_ecc=1)
    with pytest.raises(ValueError, match="mu_extent"):
        grid_values_2d_prf_polar(stimulus_centered, mu_extent=0.0)
    with pytest.raises(ValueError, match="num_angle"):
        grid_values_2d_prf_polar(stimulus_centered, num_angle=0)
    for ecc_range in [(0.0, 1.0), (2.0, 1.0)]:
        with pytest.raises(ValueError, match="smallest eccentricity"):
            grid_values_2d_prf_polar(stimulus_centered, ecc_range=ecc_range)

@pytest.mark.parametrize("num_ecc", [2, 40])
def test_grid_values_2d_prf_polar_smallest_ecc_fallback(stimulus_centered: PRFStimulus, num_ecc: int):
    """Test that the smallest ring is one grid spacing if centers there are already far enough apart or cannot be."""
    ecc = grid_values_2d_prf_polar(stimulus_centered, num_ecc=num_ecc)["ecc"]
    np.testing.assert_allclose(ecc[0], 0.5)
