"""Test Gaussian model classes."""

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.num_regression import NumericRegressionFixture
from scipy import stats
from prfmodel.models.base import BaseImpulse
from prfmodel.models.base import BaseTemporal
from prfmodel.models.base import BatchDimensionError
from prfmodel.models.base import ShapeError
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.models.gaussian import Gaussian2DResponse
from prfmodel.models.gaussian import GridMuDimensionsError
from prfmodel.models.gaussian import _check_gaussian_args
from prfmodel.models.gaussian import _expand_gaussian_args
from prfmodel.models.gaussian import predict_gaussian_response
from prfmodel.models.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.temporal import BaselineAmplitude
from prfmodel.stimulus import GridDimensionsError
from prfmodel.stimulus import Stimulus
from tests.conftest import StimulusSetup
from .conftest import parametrize_dtype


class TestCheckGaussianArgs:
    """Tests for _check_gaussian_args function."""

    def test_grid_dimensions_error(self):
        """Test that GridDimensionsError is raised."""
        grid = np.ones((4, 5, 1))  # len(shape[:-1]) = 2, shape[-1] = 1
        mu = np.ones((3, 1))
        sigma = np.ones((3, 1))
        with pytest.raises(GridDimensionsError):
            _check_gaussian_args(grid, mu, sigma)

    def test_grid_mu_dimensions_error(self):
        """Test that GridMuDimensionsError is raised."""
        grid = np.ones((4, 5, 2))
        mu = np.ones((3, 3))  # mu.shape[-1] = 3, grid.shape[-1] = 2
        sigma = np.ones((3, 1))
        with pytest.raises(GridMuDimensionsError):
            _check_gaussian_args(grid, mu, sigma)

    def test_parameter_size_error(self):
        """Test that BatchDimensionError is raised."""
        grid = np.ones((4, 5, 2))
        mu = np.ones((2, 2))
        sigma = np.ones((3, 1))  # Mismatch in first axis
        with pytest.raises(BatchDimensionError):
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


class TestSetup:
    """Setup parameters and objects for testing."""

    width: int = 5
    height: int = 4
    depth: int = 3

    @pytest.fixture
    def grid_1d(self):
        """1D stimulus grid."""
        return np.expand_dims(np.linspace(-2, 2, num=self.height), axis=1)  # (height, 1)

    @pytest.fixture
    def grid_2d(self):
        """2D stimulus grid."""
        y = np.linspace(-2, 2, num=self.height)
        x = np.linspace(-2, 2, num=self.width)
        xv, yv = np.meshgrid(x, y)
        return np.stack((xv, yv), axis=-1)  # (height, width, 2)

    @pytest.fixture
    def grid_3d(self):
        """3D stimulus grid."""
        y = np.linspace(-2, 2, num=self.height)
        x = np.linspace(-2, 2, num=self.width)
        z = np.linspace(-2, 2, num=self.depth)
        xv, yv, zv = np.meshgrid(x, y, z)
        return np.stack((xv, yv, zv), axis=-1)  # (height, width, depth, 3)

    @pytest.fixture
    def mu_1d(self):
        """1D Gaussian mu parameters."""
        return np.expand_dims(np.array([0.0, 1.0, 2.0]), axis=1)  # (num_voxels, 1)

    @pytest.fixture
    def mu_2d(self):
        """2D Gaussian mu parameters."""
        return np.array(
            [  # (num_voxels, 2)
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
        )

    @pytest.fixture
    def mu_3d(self):
        """3D Gaussian mu parameters."""
        return np.array(
            [  # (num_voxels, 3)
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
        )

    @pytest.fixture
    def sigma(self):
        """Gaussian sigma parameters."""
        return np.expand_dims(np.array([1.0, 1.5, 2.0]), axis=1)  # (num_voxels, 1)


class TestExpandGaussianArgs(TestSetup):
    """Tests for _expand_gaussian_args function."""

    @staticmethod
    def _check_shapes(grid: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        assert len(grid.shape) == len(mu.shape)
        assert len(mu.shape) - 1 == len(sigma.shape)
        assert grid.shape[-1] == mu.shape[-1]

    def test_expand_gaussian_args_1d(self, grid_1d: np.ndarray, mu_1d: np.ndarray, sigma: np.ndarray):
        """Test that 1D args are correctly expanded."""
        grid, mu, sigma = _expand_gaussian_args(grid_1d, mu_1d, sigma)

        self._check_shapes(grid, mu, sigma)

    def test_expand_gaussian_args_2d(self, grid_2d: np.ndarray, mu_2d: np.ndarray, sigma: np.ndarray):
        """Test that 2D args are correctly expanded."""
        grid, mu, sigma = _expand_gaussian_args(grid_2d, mu_2d, sigma)

        self._check_shapes(grid, mu, sigma)

    def test_expand_gaussian_args_3d(self, grid_3d: np.ndarray, mu_3d: np.ndarray, sigma: np.ndarray):
        """Test that 3D args are correctly expanded."""
        grid, mu, sigma = _expand_gaussian_args(grid_3d, mu_3d, sigma)

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

    def test_predict_gaussian_response_1d(self, grid_1d: np.ndarray, mu_1d: np.ndarray, sigma: np.ndarray):
        """Test that 1D response prediction returns correct result."""
        preds = np.asarray(predict_gaussian_response(grid_1d, mu_1d, sigma))

        assert preds.shape == (3, self.height)
        self._validate_gaussian(preds, grid_1d, mu_1d, sigma)

    def test_predict_gaussian_response_2d(self, grid_2d: np.ndarray, mu_2d: np.ndarray, sigma: np.ndarray):
        """Test that 2D response prediction returns correct result."""
        preds = np.asarray(predict_gaussian_response(grid_2d, mu_2d, sigma))

        assert preds.shape == (3, self.height, self.width)
        self._validate_gaussian(preds, grid_2d, mu_2d, sigma)

    def test_predict_gaussian_response_3d(self, grid_3d: np.ndarray, mu_3d: np.ndarray, sigma: np.ndarray):
        """Test that 3D response prediction returns correct result."""
        preds = np.asarray(predict_gaussian_response(grid_3d, mu_3d, sigma))

        assert preds.shape == (3, self.height, self.width, self.depth)
        self._validate_gaussian(preds, grid_3d, mu_3d, sigma)


class TestGaussian2DResponse(StimulusSetup):
    """Tests for Gaussian2DResponse class."""

    @pytest.fixture
    def response_model(self):
        """Response model object."""
        return Gaussian2DResponse()

    def test_parameter_names(self, response_model: Gaussian2DResponse):
        """Test that correct parameter names are returned."""
        # Order of parameter names does not matter
        assert set(response_model.parameter_names) & {"mu_y", "mu_x", "sigma"}

    @parametrize_dtype
    def test_predict(self, response_model: Gaussian2DResponse, stimulus: Stimulus, dtype: str):
        """Test that response prediction returns correct shape."""
        # 3 voxels
        params = pd.DataFrame(
            {
                "mu_x": [0.0, 1.0, 0.0],
                "mu_y": [1.0, 0.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
            },
        )

        preds = np.asarray(response_model(stimulus, params, dtype))

        # Check result shape (num_voxels, height, width)
        assert preds.shape == (params.shape[0], stimulus.design.shape[1], stimulus.design.shape[2])


class TestGaussian2DPRFModel(TestGaussian2DResponse):
    """Tests for the Gaussian2DPRFModel class."""

    @pytest.fixture
    def prf_model(self):
        """PRF model object."""
        return Gaussian2DPRFModel()

    @pytest.fixture
    def impulse_model(self):
        """Impulse response model object."""
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
            Gaussian2DPRFModel(temporal_model="test")

    def test_parameter_names(
        self,
        prf_model: Gaussian2DPRFModel,
        impulse_model: DerivativeTwoGammaImpulse,
        temporal_model: BaselineAmplitude,
        response_model: Gaussian2DResponse,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        param_names = response_model.parameter_names
        param_names.extend(impulse_model.parameter_names)
        param_names.extend(temporal_model.parameter_names)

        assert prf_model.parameter_names == list(set(param_names))

    @pytest.mark.parametrize(
        ("impulse_model", "temporal_model"),
        [
            (DerivativeTwoGammaImpulse(), BaselineAmplitude()),  # Test with class instances
            (DerivativeTwoGammaImpulse, BaselineAmplitude),  # Test with classes
            (DerivativeTwoGammaImpulse(), None),
            (DerivativeTwoGammaImpulse, None),
            (None, BaselineAmplitude()),
            (None, BaselineAmplitude),
            (None, None),
        ],
    )
    def test_predict(
        self,
        impulse_model: BaseImpulse,
        temporal_model: BaseTemporal,
        stimulus: Stimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape.

        Tests model prediction shape for both classes and class instances. Does not perform regression tests because
        predictions should be identical for classes and class instances, creating more reference files than necessary.
        Instead we perform regression tests in a separate test.

        """
        prf_model = Gaussian2DPRFModel(
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.design.shape[0])

    @pytest.mark.parametrize(
        ("impulse_model", "temporal_model"),
        [
            (DerivativeTwoGammaImpulse(), BaselineAmplitude()),  # Test with class instances
            (DerivativeTwoGammaImpulse(), None),
            (None, BaselineAmplitude()),
            (None, None),
        ],
    )
    def test_predict_regression(
        self,
        num_regression: NumericRegressionFixture,
        impulse_model: BaseImpulse,
        temporal_model: BaseTemporal,
        stimulus: Stimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference file."""
        prf_model = Gaussian2DPRFModel(
            impulse_model=impulse_model,
            temporal_model=temporal_model,
        )

        resp = prf_model(stimulus, params)

        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4},
        )
