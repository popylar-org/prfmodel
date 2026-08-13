"""Test Gaussian pRF model classes."""

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.num_regression import NumericRegressionFixture
from scipy import stats
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.exceptions import ShapeError
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse.base import BaseImpulse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.models.prf import Gaussian1DPRFModel
from prfmodel.models.prf import Gaussian1DPRFResponse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.models.prf import Gaussian2DPRFResponse
from prfmodel.models.prf import PRFStimulusEncoder
from prfmodel.models.prf import predict_gaussian_response
from prfmodel.models.prf._gaussian import _check_gaussian_args
from prfmodel.models.prf._gaussian import _expand_gaussian_args
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseScaling
from prfmodel.stimuli import PRFStimulus
from tests.conftest import PRFStimulusSetup
from tests.models.conftest import PRFStimulusGridSetup
from tests.models.conftest import parametrize_dtype


class TestCheckGaussianArgs:
    """Tests for _check_gaussian_args function."""

    def test_grid_dimensions_error(self):
        """Test that ValueError is raised when grid axis count doesn't match last dim."""
        grid = np.ones((4, 5, 1))  # len(shape[:-1]) = 2, shape[-1] = 1
        mu = np.ones((3, 1))
        sigma = np.ones((3, 1))
        with pytest.raises(ValueError, match="Number of grid axes"):
            _check_gaussian_args(grid, mu, sigma)

    def test_grid_mu_dimensions_error(self):
        """Test that ShapeMismatchError is raised."""
        grid = np.ones((4, 5, 2))
        mu = np.ones((3, 3))  # mu.shape[-1] = 3, grid.shape[-1] = 2
        sigma = np.ones((3, 1))
        with pytest.raises(ShapeMismatchError):
            _check_gaussian_args(grid, mu, sigma)

    def test_parameter_size_error(self):
        """Test that ShapeMismatchError is raised when batch dimensions of mu and sigma differ."""
        grid = np.ones((4, 5, 2))
        mu = np.ones((2, 2))
        sigma = np.ones((3, 1))  # Mismatch in first axis
        with pytest.raises(ShapeMismatchError):
            _check_gaussian_args(grid, mu, sigma)

    def test_parameter_shape_error(self):
        """Test that ParameterShapeError is raised."""
        grid = np.ones((4, 1))
        mu = np.ones(1)  # Less than two dimensions
        sigma = np.ones((3, 1))
        with pytest.raises(ShapeError):
            _check_gaussian_args(grid, mu, sigma)

        mu = np.ones((3, 1))
        sigma = np.ones(3)  # Less than two dimensions

        with pytest.raises(ShapeError):
            _check_gaussian_args(grid, mu, sigma)


class TestSetup(PRFStimulusGridSetup):
    """Setup parameters and objects for testing."""

    @pytest.fixture
    def mu(self, dim: str):
        """Gaussian mu parameters for 1D, 2D, and 3D cases."""
        if dim == "1d":
            return np.expand_dims(np.array([0.0, 1.0, 2.0]), axis=1)
        if dim == "2d":
            return np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])

    @pytest.fixture
    def sigma(self):
        """Gaussian sigma parameters."""
        return np.expand_dims(np.array([1.0, 1.5, 2.0]), axis=1)  # (num_units, 1)


class TestExpandGaussianArgs(TestSetup):
    """Tests for _expand_gaussian_args function."""

    @staticmethod
    def _check_shapes(grid: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        assert len(grid.shape) == len(mu.shape)
        assert len(mu.shape) - 1 == len(sigma.shape)
        assert grid.shape[-1] == mu.shape[-1]

    def test_expand_gaussian_args(self, grid: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        """Test that args are correctly expanded for 1D, 2D, and 3D cases."""
        grid, mu, sigma = _expand_gaussian_args(grid, mu, sigma)

        self._check_shapes(grid, mu, sigma)


class TestPredictGaussianResponse(TestSetup):
    """Tests for predict_gaussian_response function."""

    @staticmethod
    def _validate_gaussian(predictions: np.ndarray, grid: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        """Validate the predicted Gaussian response against a reference.

        Compares the predicted Gaussian response against the response from a multivariate
        Gaussian in `scipy.stats`.

        """
        expected = np.stack(
            [
                stats.multivariate_normal.pdf(grid, mean=mu[i], cov=sigma[i, 0] ** 2 * np.eye(grid.shape[-1]))
                for i in range(mu.shape[0])
            ],
        )
        assert np.allclose(predictions, expected)

    def test_predict_gaussian_response(self, grid: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
        """Test that response prediction returns correct result for 1D, 2D, and 3D cases."""
        preds = np.asarray(predict_gaussian_response(grid, mu, sigma))

        assert preds.shape == (mu.shape[0], *grid.shape[:-1])
        self._validate_gaussian(preds, grid, mu, sigma)


class TestGaussian2DPRFResponse(PRFStimulusSetup):
    """Tests for Gaussian2DResponse class."""

    @pytest.fixture
    def response_model(self):
        """Response model object."""
        return Gaussian2DPRFResponse()

    def test_parameter_names(self, response_model: Gaussian2DPRFResponse):
        """Test that correct parameter names are returned."""
        assert response_model.parameter_names == ["mu_y", "mu_x", "sigma"]

    @parametrize_dtype
    def test_predict(self, response_model: Gaussian2DPRFResponse, stimulus: PRFStimulus, dtype: str):
        """Test that response prediction returns correct shape."""
        # 3 units
        params = pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
            },
        )

        preds = np.asarray(response_model(stimulus, params, dtype))

        # Check result shape (num_units, height, width)
        assert preds.shape == (params.shape[0], stimulus.design.shape[1], stimulus.design.shape[2])


class TestGaussian1DPRFResponse:
    """Tests for Gaussian1DResponse class."""

    @pytest.fixture
    def stimulus(self):
        """1D pRF stimulus."""
        grid = np.expand_dims(np.arange(4), 1)
        design = np.eye(4, 4)

        return PRFStimulus(
            design=design,
            grid=grid,
        )

    @pytest.fixture
    def response_model(self):
        """Response model object."""
        return Gaussian1DPRFResponse()

    def test_parameter_names(self, response_model: Gaussian1DPRFResponse):
        """Test that correct parameter names are returned."""
        assert response_model.parameter_names == ["mu", "sigma"]

    @parametrize_dtype
    def test_predict(self, response_model: Gaussian1DPRFResponse, stimulus: PRFStimulus, dtype: str):
        """Test that response prediction returns correct shape."""
        # 3 units
        params = pd.DataFrame(
            {
                "mu": [0.0, 1.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
            },
        )

        preds = np.asarray(response_model(stimulus, params, dtype))

        # Check result shape (num_units, num_coordinates)
        assert preds.shape == (params.shape[0], stimulus.design.shape[1])


class TestGaussian2DPRFModel(TestGaussian2DPRFResponse):
    """Tests for the Gaussian2DPRFModel class."""

    @pytest.fixture
    def prf_model(self):
        """PRF model object."""
        return Gaussian2DPRFModel()

    @pytest.fixture
    def impulse_model(self):
        """Impulse model object."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def temporal_model(self):
        """Temporal model object."""
        return BaselineAmplitude()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "baseline": [0.0, 0.1, 0.2],
                "amplitude": [1.1, 1.0, 0.9],
            },
        )

    def test_submodels_inherit_basemodel(self):
        """Test that submodels that do not inherit from BaseModel raise an error."""
        with pytest.raises(TypeError):
            Gaussian2DPRFModel(impulse_model="test")

        with pytest.raises(TypeError):
            Gaussian2DPRFModel(scaling_model="test")

    def test_parameter_names(
        self,
        prf_model: Gaussian2DPRFModel,
        impulse_model: DerivativeTwoGammaImpulse,
        temporal_model: BaselineAmplitude,
        response_model: Gaussian2DPRFResponse,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        param_names = response_model.parameter_names
        param_names.extend(impulse_model.parameter_names)
        param_names.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(dict.fromkeys(param_names))

    @pytest.mark.parametrize(
        "temporal_model",
        [None, BaselineAmplitude, BaselineAmplitude()],
    )
    @pytest.mark.parametrize(
        "impulse_model",
        [None, DerivativeTwoGammaImpulse, DerivativeTwoGammaImpulse()],
    )
    @pytest.mark.parametrize("encoding_model", [PRFStimulusEncoder, PRFStimulusEncoder()])
    def test_predict(
        self,
        encoding_model: BaseStimulusEncoder,
        impulse_model: BaseImpulse,
        temporal_model: BaseScaling,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape.

        Tests model prediction shape for both classes and class instances. Does not perform regression tests because
        predictions should be identical for classes and class instances, creating more reference files than necessary.
        Instead we perform regression tests in a separate test.

        """
        prf_model = Gaussian2DPRFModel(
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    @pytest.mark.parametrize(
        "temporal_model",
        [None, BaselineAmplitude()],
    )
    @pytest.mark.parametrize(
        "impulse_model",
        [None, DerivativeTwoGammaImpulse()],
    )
    def test_predict_regression(
        self,
        num_regression: NumericRegressionFixture,
        impulse_model: BaseImpulse,
        temporal_model: BaseScaling,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference file."""
        prf_model = Gaussian2DPRFModel(
            impulse_model=impulse_model,
            scaling_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        # `rtol` matches `test_div_norm` and `test_delayed_norm`. Without it, numpy's 1e-5 default
        # applies, and the budget `atol + rtol * |value|` collapses to ~1e-4 wherever the response
        # crosses near zero -- while the float32 error there is set by the series peak (~150 for
        # these parameters), not by the local value, because the convolution mixes the frames. That
        # left frame 23 of unit 0 at 88% of its budget, and Windows' libm put it over.
        # This costs no detection power: the sampling bug these snapshots were regenerated for moves
        # unit 0 by up to 40.8, and trips the same 44 of 50 frames at either rtol.
        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4, "rtol": 1e-3},
        )


class TestGaussian1DPRFModel(TestGaussian1DPRFResponse):
    """Tests for the Gaussian1DPRFModel class.

    Does not include regression tests because the class uses the same underlying functions as Gaussian2DPRFModel.

    """

    @pytest.fixture
    def prf_model(self):
        """PRF model object."""
        return Gaussian1DPRFModel()

    @pytest.fixture
    def impulse_model(self):
        """Impulse model object."""
        return DerivativeTwoGammaImpulse()

    @pytest.fixture
    def temporal_model(self):
        """Temporal model object."""
        return BaselineAmplitude()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "mu": [0.0, 1.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "delay": [6.0, 7.0, 5.0],
                "dispersion": [0.9, 1.0, 0.8],
                "undershoot": [12.0, 11.0, 13.0],
                "u_dispersion": [0.9, 1.0, 0.8],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "baseline": [0.0, 0.1, 0.2],
                "amplitude": [1.1, 1.0, 0.9],
            },
        )

    def test_parameter_names(
        self,
        prf_model: Gaussian1DPRFModel,
        impulse_model: DerivativeTwoGammaImpulse,
        temporal_model: BaselineAmplitude,
        response_model: Gaussian1DPRFResponse,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        param_names = response_model.parameter_names
        param_names.extend(impulse_model.parameter_names)
        param_names.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(dict.fromkeys(param_names))

    @pytest.mark.parametrize(
        "temporal_model",
        [None, BaselineAmplitude, BaselineAmplitude()],
    )
    @pytest.mark.parametrize(
        "impulse_model",
        [None, DerivativeTwoGammaImpulse, DerivativeTwoGammaImpulse()],
    )
    @pytest.mark.parametrize("encoding_model", [PRFStimulusEncoder, PRFStimulusEncoder()])
    def test_predict(
        self,
        encoding_model: BaseStimulusEncoder,
        impulse_model: BaseImpulse,
        temporal_model: BaseScaling,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape.

        Tests model prediction shape for both classes and class instances.

        """
        prf_model = Gaussian1DPRFModel(
            encoding_model=encoding_model,
            impulse_model=impulse_model,
            scaling_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])


class TestCoordinateConvention:
    """Tests that `mu_x` and `mu_y` map onto the stimulus axes they are named after.

    These tests deliberately avoid the simulate-with-the-model-then-fit-the-same-model pattern used
    elsewhere: a swap of the two centre parameters is self-consistent and would pass such a test.
    Instead they place stimulus energy at an independently known location and check which parameter
    has to move to follow it.

    """

    num_pixels: int = 64
    pixel_size: float = 0.1
    bar_position: float = 2.0

    # A pRF centred on the bar integrates a large fraction of it; one centred off it sees ~nothing
    on_bar_min_response: float = 1.0
    off_bar_max_response: float = 1e-3

    def _make_bar_stimulus(self, axis: str) -> tuple[PRFStimulus, float]:
        """Build a static bar at a known coordinate along `axis`, uniform along the other axis."""
        size = self.num_pixels
        coords = (np.arange(size) - (size - 1) / 2) * self.pixel_size
        xv, yv = np.meshgrid(coords, coords)
        grid = np.stack((yv, xv), axis=-1)

        index = int(np.argmin(np.abs(coords - self.bar_position)))
        design = np.zeros((4, size, size))

        if axis == "x":
            # A vertical bar at a fixed x, spanning every y
            design[:, :, index - 2 : index + 3] = 1.0
        else:
            # A horizontal bar at a fixed y, spanning every x
            design[:, index - 2 : index + 3, :] = 1.0

        return PRFStimulus(design=design, grid=grid, dimension_labels=["y", "x"]), coords[index]

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_prf_responds_on_the_axis_it_is_named_after(self, axis: str):
        """Test that a pRF offset along one axis only sees a bar placed on that same axis."""
        stimulus, position = self._make_bar_stimulus(axis)

        on_bar = {"mu_x": 0.0, "mu_y": 0.0, "sigma": 0.3}
        on_bar[f"mu_{axis}"] = position

        off_bar = {"mu_x": on_bar["mu_y"], "mu_y": on_bar["mu_x"], "sigma": 0.3}

        params = pd.DataFrame([on_bar, off_bar])
        resp = np.asarray(Gaussian2DPRFResponse()(stimulus, params))
        encoded = np.asarray(PRFStimulusEncoder()(stimulus, resp, params))

        assert encoded[0, 0] > self.on_bar_min_response, (
            f"pRF at mu_{axis}={position} does not respond to a bar on the {axis} axis; "
            f"mu_x and mu_y are swapped relative to the stimulus grid"
        )
        assert encoded[1, 0] < self.off_bar_max_response, (
            f"pRF offset along the other axis should not see a bar on the {axis} axis"
        )

    @staticmethod
    def _assert_grid_convention(grid: np.ndarray, name: str) -> None:
        """Assert the shared axis-order and axis-sign convention of the built-in stimuli."""
        # Component 0 (y) varies down the rows and is constant across columns
        assert np.allclose(np.std(grid[..., 0], axis=1), 0.0), f"{name}: grid[..., 0] must be constant along columns"
        assert grid[-1, 0, 0] > grid[0, 0, 0], f"{name}: grid[..., 0] (y) must increase down the rows"

        # Component 1 (x) varies across the columns and is constant down the rows
        assert np.allclose(np.std(grid[..., 1], axis=0), 0.0), f"{name}: grid[..., 1] must be constant along rows"
        # x decreases because the design is in screen pixel order and the display is mirrored
        assert grid[0, -1, 1] < grid[0, 0, 1], (
            f"{name}: grid[..., 1] (x) must decrease across the columns; the design is stored in "
            f"screen pixel order and the mirrored display flips the horizontal axis"
        )

    def test_bar_stimulus_grid_follows_convention(self):
        """Test that `create_2d_bar_stimulus` uses the documented axis order and sign."""
        stimulus = PRFStimulus.create_2d_bar_stimulus(num_frames=10, width=16, height=8, pixel_size=0.5)

        assert stimulus.grid.shape == (8, 16, 2)
        self._assert_grid_convention(stimulus.grid, "create_2d_bar_stimulus")

    def test_packaged_example_grid_follows_convention(self):
        """Test that the packaged example stimulus uses the documented axis order and sign."""
        stimulus = load_2d_prf_bar_stimulus()

        assert stimulus.grid.shape[:-1] == stimulus.design.shape[1:]
        self._assert_grid_convention(stimulus.grid, "load_2d_prf_bar_stimulus")

    def test_built_in_stimuli_agree_on_hemifield(self):
        """Test that a bar in the leftmost columns falls in the same hemifield for both stimuli.

        This is the property that makes `mu_x` estimates comparable across the two built-in
        stimulus sources. It fails if either one flips its horizontal axis independently.

        """
        generated = PRFStimulus.create_2d_bar_stimulus(num_frames=10, width=32, height=32)
        packaged = load_2d_prf_bar_stimulus()

        for name, stimulus in [("create_2d_bar_stimulus", generated), ("load_2d_prf_bar_stimulus", packaged)]:
            # x coordinate of the leftmost and rightmost design column
            leftmost = stimulus.grid[0, 0, 1]
            rightmost = stimulus.grid[0, -1, 1]

            assert leftmost > 0.0 > rightmost, (
                f"{name}: the leftmost design column must map to the right visual hemifield "
                f"(positive x) and the rightmost column to the left hemifield"
            )
