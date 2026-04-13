"""Test Gaussian CF model classes."""

import numpy as np
import pandas as pd
import pytest
from pytest_regressions.num_regression import NumericRegressionFixture
from prfmodel.models.base import BaseEncoder
from prfmodel.models.cf.gaussian import GaussianCFModel
from prfmodel.models.cf.gaussian import GaussianCFResponse
from prfmodel.models.cf.stimulus_encoding import CFStimulusEncoder
from prfmodel.scaling import BaselineAmplitude
from prfmodel.scaling.base import BaseTemporal
from prfmodel.stimuli.cf import CFStimulus
from tests.models.conftest import CFSetup
from tests.models.conftest import parametrize_dtype


class TestGaussianCFResponse(CFSetup):
    """Tests for GaussianCFResponse class."""

    @pytest.fixture
    def response_model(self):
        """Response model object."""
        return GaussianCFResponse()

    def test_parameter_names(self, response_model: GaussianCFResponse):
        """Test that correct parameter names are returned."""
        # Order of parameter names does not matter
        assert set(response_model.parameter_names) & {"center_index", "sigma"}

    @parametrize_dtype
    def test_predict(self, response_model: GaussianCFResponse, stimulus: CFStimulus, dtype: str):
        """Test that response prediction returns correct shape."""
        # 3 units
        params = pd.DataFrame(
            {
                "center_index": [0, 1, 2],
                "sigma": [1.0, 2.0, 3.0],
            },
        )

        preds = np.asarray(response_model(stimulus, params, dtype))

        # Check result shape
        assert preds.shape == (params.shape[0], self.num_source)  # (num_units, distance_matrix.shape[0])


class TestGaussianCFModel(TestGaussianCFResponse):
    """Tests for the GaussianCFModel class."""

    @pytest.fixture
    def cf_model(self):
        """CF model object."""
        return GaussianCFModel()

    @pytest.fixture
    def temporal_model(self):
        """Temporal model object."""
        return BaselineAmplitude()

    @pytest.fixture
    def params(self):
        """Dataframe with parameters."""
        return pd.DataFrame(
            {
                "center_index": [0, 2, 1],
                "sigma": [1.0, 2.0, 3.0],
                "baseline": [0.5, -0.1, 0.2],
                "amplitude": [-1.1, 0.5, 2.0],
            },
        )

    def test_submodels_inherit_basemodel(self):
        """Test that submodels that do not inherit from BaseModel raise an error."""
        with pytest.raises(TypeError):
            GaussianCFModel(temporal_model="test")

    def test_parameter_names(
        self,
        cf_model: GaussianCFModel,
        temporal_model: BaselineAmplitude,
        response_model: GaussianCFResponse,
    ):
        """Test that parameter names of composite model match parameter names of submodels."""
        param_names = response_model.parameter_names
        param_names.extend(temporal_model.parameter_names)

        assert cf_model.parameter_names == list(dict.fromkeys(param_names))

    @pytest.mark.parametrize(
        "temporal_model",
        [
            BaselineAmplitude(),  # Test with class instance
            BaselineAmplitude,  # Test with class
            None,
        ],
    )
    @pytest.mark.parametrize(
        "encoding_model",
        [CFStimulusEncoder, CFStimulusEncoder()],
    )
    def test_predict(
        self,
        encoding_model: BaseEncoder,
        temporal_model: BaseTemporal,
        stimulus: CFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction returns correct shape."""
        cf_model = GaussianCFModel(
            encoding_model=encoding_model,
            temporal_model=temporal_model,
        )

        resp = cf_model(stimulus, params)

        assert resp.shape == (params.shape[0], stimulus.source_response.shape[-1])

    def test_predict_regression_cf(
        self,
        num_regression: NumericRegressionFixture,
        stimulus: CFStimulus,
        params: pd.DataFrame,
    ):
        """Test that model prediction matches reference file."""
        cf_model = GaussianCFModel()

        resp = cf_model(stimulus, params)

        num_regression.check(
            {f"response_{i}": x for i, x in enumerate(resp)},
            default_tolerance={"atol": 1e-4},
        )
