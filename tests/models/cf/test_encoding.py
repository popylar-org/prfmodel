"""Tests for CF encoder classes."""

import numpy as np
import pandas as pd
import pytest
from prfmodel.models.cf import CFStimulusEncoder
from prfmodel.models.compression import CompressiveEncoder
from prfmodel.stimuli import CFStimulus
from tests.models.conftest import parametrize_dtype


class TestSetup:
    """Setup for CF encoder tests."""

    num_units: int = 3
    num_source: int = 6
    num_frames: int = 10

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
class TestCompressiveEncoderCF(TestSetup):
    """Tests for CompressiveEncoder with CF stimulus."""

    @pytest.fixture
    def parameters(self):
        """Compressive encoder parameters."""
        return pd.DataFrame(
            {
                "gain": [1.2, 0.9, 0.1],
                "n": [0.8, 1.2, 0.5],
            },
        )

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
