"""Tests for encoder classes and methods."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.encoding import CFStimulusEncoder
from prfmodel.models.encoding import CompressiveEncoder
from prfmodel.models.encoding import PRFStimulusEncoder
from prfmodel.models.encoding import encode_prf_response
from prfmodel.stimuli.cf import CFStimulus
from prfmodel.stimuli.prf import PRFStimulus
from .conftest import PRFStimulusGridSetup
from .conftest import parametrize_dtype


class TestSetup(PRFStimulusGridSetup):
    """Setup for tests."""

    num_units: int = 3
    num_source: int = 6
    num_frames: int = 10

    @pytest.fixture
    def response_1d(self):
        """1D model response."""
        resp_dummy = np.ones((self.num_units, self.num_height))
        # Response is function of height
        return resp_dummy * np.expand_dims(np.sin(np.arange(self.num_height)), 0)

    @pytest.fixture
    def response_2d(self):
        """2D model response."""
        resp_dummy = np.ones((self.num_units, self.num_height, self.num_width))
        # Response is function of width
        return resp_dummy * np.expand_dims(np.sin(np.arange(self.num_width)), (0, 1))

    @pytest.fixture
    def response_3d(self):
        """3D model response."""
        resp_dummy = np.ones((self.num_units, self.num_height, self.num_width, self.num_depth))
        # Response is function of depth
        return resp_dummy * np.expand_dims(np.sin(np.arange(self.num_depth)), (0, 1, 2))

    @pytest.fixture
    def cf_response(self):
        """CF model response of shape (num_units, num_vertices)."""
        return np.ones((self.num_units, self.num_source)) * np.expand_dims(np.sin(np.arange(self.num_source)), 0)

    @pytest.fixture
    def cf_stimulus(self):
        """CF stimulus with an identity distance matrix and a source response varying over vertices."""
        distance_matrix = np.eye(self.num_source)
        source_response = np.ones((self.num_source, self.num_frames)) * np.expand_dims(
            np.arange(self.num_source),
            1,
        )
        return CFStimulus(distance_matrix, source_response)


@parametrize_dtype
class TestEncodePRFResponse(TestSetup):
    """Tests for encode_prf_response."""

    def _check_encoding(self, x: np.ndarray) -> None:
        assert x.shape == (self.num_units, self.num_frames)
        # Check that all units have identical encoding for identical response and design
        assert np.unique(x, axis=1).shape == (self.num_units, 1)

    def test_call_1d(self, response_1d: np.ndarray, dtype: str):
        """Test that 1D encoding returns the correct shape."""
        design = np.ones((self.num_frames, self.num_height))

        x = np.asarray(encode_prf_response(response_1d, design, dtype))

        self._check_encoding(x)

    def test_call_2d(self, response_2d: np.ndarray, dtype: str):
        """Test that 2D encoding returns the correct shape."""
        design = np.ones((self.num_frames, self.num_height, self.num_width))

        x = np.asarray(encode_prf_response(response_2d, design, dtype))

        self._check_encoding(x)

    def test_call_3d(self, response_3d: np.ndarray, dtype: str):
        """Test that 3D encoding returns the correct shape."""
        design = np.ones((self.num_frames, self.num_height, self.num_width, self.num_depth))

        x = np.asarray(encode_prf_response(response_3d, design, dtype))

        self._check_encoding(x)


@parametrize_dtype
class TestPRFStimulusEncoder(TestSetup):
    """Tests for PRFStimulusEncoder class."""

    @pytest.fixture
    def prf_stimulus_encoder(self):
        """Stimulus encoder object."""
        return PRFStimulusEncoder()

    @pytest.fixture
    def parameters(self):
        """Stimulus encoder parameters (empty because no parameters are required)."""
        return pd.DataFrame()

    def test_call_1d(
        self,
        response_1d: np.ndarray,
        grid_1d: np.ndarray,
        parameters: pd.DataFrame,
        prf_stimulus_encoder: PRFStimulusEncoder,
        dtype: str,
    ):
        """Test that 1D encoding returns the same result as encode_prf_response."""
        design = np.ones((self.num_frames, self.num_height))
        stimulus = PRFStimulus(design, grid_1d)

        response_encoded_ref = np.asarray(encode_prf_response(response_1d, design, dtype))
        response_encoded = np.asarray(prf_stimulus_encoder(stimulus, response_1d, parameters, dtype))

        assert np.array_equal(response_encoded_ref, response_encoded)

    def test_call_2d(
        self,
        response_2d: np.ndarray,
        grid_2d: np.ndarray,
        parameters: pd.DataFrame,
        prf_stimulus_encoder: PRFStimulusEncoder,
        dtype: str,
    ):
        """Test that 2D encoding returns the same result as encode_prf_response."""
        design = np.ones((self.num_frames, self.num_height, self.num_width))
        stimulus = PRFStimulus(design, grid_2d)

        response_encoded_ref = np.asarray(encode_prf_response(response_2d, design, dtype))
        response_encoded = np.asarray(prf_stimulus_encoder(stimulus, response_2d, parameters, dtype))

        assert np.array_equal(response_encoded_ref, response_encoded)

    def test_call_3d(
        self,
        response_3d: np.ndarray,
        grid_3d: np.ndarray,
        parameters: pd.DataFrame,
        prf_stimulus_encoder: PRFStimulusEncoder,
        dtype: str,
    ):
        """Test that 3D encoding returns the same result as encode_prf_response."""
        design = np.ones((self.num_frames, self.num_height, self.num_width, self.num_depth))
        stimulus = PRFStimulus(design, grid_3d)

        response_encoded_ref = np.asarray(encode_prf_response(response_3d, design, dtype))
        response_encoded = np.asarray(prf_stimulus_encoder(stimulus, response_3d, parameters, dtype))

        assert np.array_equal(response_encoded_ref, response_encoded)


class TestCFStimulusEncoder(TestSetup):
    """Tests for CFStimulusEncoder class."""

    @pytest.fixture
    def cf_stimulus_encoder(self):
        """Stimulus encoder object."""
        return CFStimulusEncoder()

    @pytest.fixture
    def parameters(self):
        """Stimulus encoder parameters (empty because no parameters are required)."""
        return pd.DataFrame()

    @parametrize_dtype
    def test_call_shape(
        self,
        cf_response: np.ndarray,
        cf_stimulus: CFStimulus,
        parameters: pd.DataFrame,
        cf_stimulus_encoder: CFStimulusEncoder,
        dtype: str,
    ):
        """Test that CF encoding returns shape (num_units, num_frames)."""
        result = np.asarray(cf_stimulus_encoder(cf_stimulus, cf_response, parameters, dtype))

        assert result.shape == (self.num_units, self.num_frames)

    @parametrize_dtype
    def test_call_values(
        self,
        cf_response: np.ndarray,
        cf_stimulus: CFStimulus,
        parameters: pd.DataFrame,
        cf_stimulus_encoder: CFStimulusEncoder,
        dtype: str,
    ):
        """Test that CF encoding matches manual matrix multiplication over vertices."""
        result = np.asarray(cf_stimulus_encoder(cf_stimulus, cf_response, parameters, dtype))
        expected = cf_response @ cf_stimulus.source_response

        # Encoder uses tensordot operation which leads to slight numerical differences
        assert np.allclose(result, expected, atol=1e-5)

    def test_response_source_dim_mismatch_error(
        self,
        cf_stimulus: CFStimulus,
        parameters: pd.DataFrame,
        cf_stimulus_encoder: CFStimulusEncoder,
    ):
        """Test that dimension mismatches between connective field and source response raise an error."""
        cf_response = np.ones((self.num_units, 4))

        with pytest.raises(ValueError):
            cf_stimulus_encoder(cf_stimulus, cf_response, parameters)


@parametrize_dtype
class TestCompressiveEncoder(TestSetup):
    """Tests for CompressiveEncoder class."""

    @pytest.fixture
    def parameters(self):
        """Compressive encoder parameters."""
        return pd.DataFrame(
            {
                "gain": [1.2, 0.9, 0.1],
                "n": [0.8, 1.2, 0.5],
            },
        )

    def test_call_shape_prf_stimulus_encoder(
        self,
        response_2d: np.ndarray,
        grid_2d: np.ndarray,
        parameters: pd.DataFrame,
        dtype: str,
    ):
        """Test that compressive encoding for PRFStimulus returns same shape as normal encoding."""
        design = np.ones((self.num_frames, self.num_height, self.num_width))
        stimulus = PRFStimulus(design, grid_2d)

        encoder = PRFStimulusEncoder()
        compressive_encoder = CompressiveEncoder(encoding_model=encoder)

        encoded_response = np.asarray(encoder(stimulus, response_2d, pd.DataFrame(), dtype))
        compressed_response = np.asarray(compressive_encoder(stimulus, response_2d, parameters, dtype))

        assert encoded_response.shape == compressed_response.shape

    def test_call_shape_cf_stimulus_encoder(
        self,
        cf_stimulus: CFStimulus,
        cf_response: np.ndarray,
        parameters: pd.DataFrame,
        dtype: str,
    ):
        """Test that compressive encoding for CFStimulus returns same shape as normal encoding."""
        encoder = CFStimulusEncoder()
        compressive_encoder = CompressiveEncoder(encoding_model=encoder)

        encoded_response = np.asarray(encoder(cf_stimulus, cf_response, pd.DataFrame(), dtype))
        compressed_response = np.asarray(compressive_encoder(cf_stimulus, cf_response, parameters, dtype))

        assert encoded_response.shape == compressed_response.shape
