"""Setup for model tests."""

import numpy as np
import pytest
from prfmodel.stimuli.cf import CFStimulus

parametrize_dtype = pytest.mark.parametrize("dtype", [None, "float32", "float64"])


def _make_grid(  # noqa: PLR0913 (too many arguments)
    dim: str,
    num_height: int,
    num_width: int,
    num_depth: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    if dim == "1d":
        return np.expand_dims(np.linspace(lower, upper, num=num_height), axis=1)

    if dim == "2d":
        xv, yv = np.meshgrid(np.linspace(lower, upper, num=num_width), np.linspace(lower, upper, num=num_height))

        return np.stack((xv, yv), axis=-1)

    xv, yv, zv = np.meshgrid(
        np.linspace(lower, upper, num=num_width),
        np.linspace(lower, upper, num=num_height),
        np.linspace(lower, upper, num=num_depth),
    )

    return np.stack((xv, yv, zv), axis=-1)


class PRFStimulusGridSetup:
    """Test setup for pRF stimulus grids."""

    lower: float = -2.0
    upper: float = 2.0

    num_width: int = 5
    num_height: int = 4
    num_depth: int = 3

    @pytest.fixture(params=["1d", "2d", "3d"])
    def dim(self, request: pytest.FixtureRequest):
        """Dimensionality of the grid fixture."""
        return request.param

    @pytest.fixture
    def grid(self, dim: str):
        """Stimulus grid for 1D, 2D, and 3D cases."""
        return _make_grid(dim, self.num_height, self.num_width, self.num_depth, self.lower, self.upper)

    @pytest.fixture
    def grid_1d(self):
        """1D stimulus grid of shape (height, 1)."""
        return _make_grid("1d", self.num_height, self.num_width, self.num_depth, self.lower, self.upper)

    @pytest.fixture
    def grid_2d(self):
        """2D stimulus grid of shape (height, width, 2)."""
        return _make_grid("2d", self.num_height, self.num_width, self.num_depth, self.lower, self.upper)

    @pytest.fixture
    def grid_3d(self):
        """3D stimulus grid of shape (height, width, depth, 3)."""
        return _make_grid("3d", self.num_height, self.num_width, self.num_depth, self.lower, self.upper)


class CFSetup:
    """Test setup for CF models."""

    num_source: int = 5

    @pytest.fixture
    def distance_matrix(self):
        """Distance matrix."""
        return np.ones((self.num_source, self.num_source))

    @pytest.fixture
    def source_response(self):
        """Source response."""
        return np.array([[1.5, 2.0, 0.2, 3.1, 1.7]]).T  # (distance_matrix.shape[0], 1)

    @pytest.fixture
    def stimulus(self, distance_matrix: np.ndarray, source_response: np.ndarray):
        """Stimulus object."""
        return CFStimulus(
            distance_matrix=distance_matrix,
            source_response=source_response,
        )
