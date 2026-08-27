"""Tests for utility functions and classes."""

from abc import ABC
from abc import abstractmethod
import keras
import numpy as np
import pandas as pd
import pytest
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor
from prfmodel.utils import CompositeModelProtocol
from prfmodel.utils import ModelProtocol
from prfmodel.utils import TensorFrame
from prfmodel.utils import _get_norm_fun
from prfmodel.utils import batched
from prfmodel.utils import normalize_response
from .conftest import TestSetup


class _DummyModel(ModelProtocol):
    _positive_parameter_names: tuple[str, ...] = "a"

    @property
    def parameter_names(self) -> list[str]:
        return ["a"]


class _DummyModelWithDefault(ModelProtocol):
    def __init__(self):
        self.default_parameters: dict[str, float] = {"a": [1.0]}

    @property
    def parameter_names(self) -> list[str]:
        return ["a", "b"]


class _DummyCompositeModel(CompositeModelProtocol):
    def __init__(self, **models: dict[str, ModelProtocol]):
        self._models = models


class ModelProtocolTestSetup(ABC):
    """Test setup for model protocol classes."""

    @abstractmethod
    def model(self) -> ModelProtocol:
        """Create an instance of the model protocol class."""

    def test_check_parameters_names_error(self, model: ModelProtocol):
        """Test that check_parameter_names raises error on missing required parameters."""
        parameters = pd.DataFrame(
            {
                "b": [1, 2, 3],
            },
        )

        with pytest.raises(ValueError, match=r"\['a'\]"):
            model.check_parameter_names(parameters)

    def test_check_parameters_names_no_error(self, model: ModelProtocol):
        """Test that check_parameter_names raises no error when no required parameters are missing."""
        parameters = pd.DataFrame(
            {
                "a": [1, 2, 3],
            },
        )

        model.check_parameter_names(parameters)

    def test_check_parameter_values_error(self, model: ModelProtocol):
        """Test that check_parameter_values raises error on positive parameters."""
        parameters = pd.DataFrame(
            {
                "a": [-1, 1, 2],
            },
        )
        with pytest.raises(ValueError, match=r"'a'.*> 0"):
            model.check_parameter_values(parameters)

    def test_check_parameter_values_no_error(self, model: ModelProtocol):
        """Test that check_parameter_values raises no error on positive parameters."""
        parameters = pd.DataFrame(
            {
                "a": [1, 2, 3],
            },
        )

        model.check_parameter_values(parameters)


class TestModelProtocol(ModelProtocolTestSetup):
    """Tests for ModelProtocol class."""

    @pytest.fixture
    def model(self):
        """Create a dummy model instance."""
        return _DummyModel()


class TestCompositeModelProtocol(ModelProtocolTestSetup):
    """Tests for CompositeModelProtocol class."""

    @pytest.fixture
    def model(self):
        """Create a dummy composite model instance."""
        return _DummyCompositeModel(dummy=_DummyModel())

    def test_models_setter_getter(self, model: _DummyCompositeModel):
        """Test that 'models' setter and getter methods round-trip."""
        models = model.models
        model.model = models
        assert model.models == models

    def test_models_setter_errors_no_model_protocol(self, model: ModelProtocol):
        """Test that setter raise error when a model does not inherit from 'ModelProtocol'."""
        with pytest.raises(TypeError, match=r"inherit.*ModelProtocol"):
            model.models = {"dummy": object()}

    def test_get_consumed_parameter_names_omits_default(self):
        """Test that get_consumed_parameter_names omits default parameters."""
        model = _DummyCompositeModel(dummy_default=_DummyModelWithDefault())
        parameters = pd.DataFrame(
            {
                "b": [0, 1, 2],
            },
        )
        assert model.get_consumed_parameter_names(parameters) == ["b"]

    def test_get_consumed_parameter_names_keeps_overwrite(self):
        """Test that get_consumed_parameter_names does no omits default parameters when they are overwritten."""
        model = _DummyCompositeModel(dummy_default=_DummyModelWithDefault())
        parameters = pd.DataFrame(
            {
                "a": [0, 1, 2],
                "b": [0, 1, 2],
            },
        )
        assert model.get_consumed_parameter_names(parameters) == ["a", "b"]


@pytest.mark.parametrize("norm", [None, "sum", "mean", "max", "norm"])
def test_normalize_response(norm: str):
    """Test that normalize_response returns correct result."""
    response = np.expand_dims(np.linspace(-5, 5, 100), 0)
    response_norm = np.asarray(normalize_response(response, norm=norm))

    assert response_norm.shape == response.shape

    if norm is not None:
        norm_fun = _get_norm_fun(norm)
        response_norm_ref = response / np.asarray(norm_fun(response, axis=1, keepdims=True))
    else:
        response_norm_ref = response

    assert np.allclose(response_norm, response_norm_ref)


def test_normalize_response_error():
    """Test that normalize_response raises an error for wrong input shape."""
    response = np.ones((10,))

    with pytest.raises(ValueError):
        normalize_response(response)

    response = 10

    with pytest.raises(ValueError):
        normalize_response(response)

    response = np.ones((10, 2, 1))

    with pytest.raises(ValueError):
        normalize_response(response)


class TestTensorFrame:
    """Tests for TensorFrame class."""

    shape: tuple[int] = (3, 1)

    @pytest.fixture
    def tensor_frame(self):
        """TensorFrame object."""
        return TensorFrame({"a": 0.0, "b": [1.0], "c": np.ones(self.shape[0]), "d": keras.ops.ones(self.shape)})

    def test_get_item(self, tensor_frame: TensorFrame):
        """Test that getting an item with a single key returns the correct shape and values."""
        for key in tensor_frame.columns:
            x = tensor_frame[key]
            assert x.shape == self.shape[:1]

        # torch requires us to convert tensors to numpy arrays before we can compare against floats
        assert np.all(np.asarray(tensor_frame["a"]) == 0.0)
        assert np.all(np.asarray(tensor_frame["b"]) == 1.0)

    def test_get_item_list(self, tensor_frame: TensorFrame):
        """Test that getting items with a list of keys returns the correct shapes and values."""
        x = tensor_frame[tensor_frame.columns]
        assert x.shape == (self.shape[0], len(tensor_frame.columns))

    def test_set_item(self, tensor_frame: TensorFrame):
        """Test that setting an item with a single key stores the correct shape and values."""
        new_tensor_frame = TensorFrame(
            {
                "e": keras.ops.zeros(self.shape),
            },
        )

        for key in tensor_frame.columns:
            new_tensor_frame[key] = tensor_frame[key]
            x = new_tensor_frame[key]
            assert x.shape == self.shape[:1]

    def test_set_item_list(self, tensor_frame: TensorFrame):
        """Test that setting items with a list of keys stores the correct shapes and values."""
        new_tensor_frame = TensorFrame(
            {
                "e": keras.ops.zeros(self.shape),
            },
        )
        new_tensor_frame[tensor_frame.columns] = tensor_frame[tensor_frame.columns]
        x = new_tensor_frame[tensor_frame.columns]
        assert x.shape == (self.shape[0], len(tensor_frame.columns))


class TestBatched(TestSetup):
    """Tests for the batched decorator."""

    def test_batch_size_none_returns_same_result(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that batch_size=None calls the function once with all units."""
        result_unbatched = model(stimulus, params)
        result_batched = batched(model)(stimulus, params, batch_size=None)

        assert np.array_equal(np.asarray(result_batched), np.asarray(result_unbatched))

    def test_batched_matches_unbatched(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that batched results match unbatched results."""
        result_unbatched = model(stimulus, params)
        result_batched = batched(model)(stimulus, params, batch_size=3)

        assert np.allclose(np.asarray(result_batched), np.asarray(result_unbatched))

    def test_output_shape(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that the output shape is (num_units, num_frames)."""
        result = batched(model)(stimulus, params, batch_size=3)

        assert result.shape == (params.shape[0], stimulus.design.shape[0])

    def test_batch_size_larger_than_num_units(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that a batch_size larger than the number of units works."""
        result = batched(model)(stimulus, params, batch_size=100)
        expected = model(stimulus, params)

        assert np.allclose(np.asarray(result), np.asarray(expected))

    def test_exact_batch_division(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test with a batch_size that evenly divides num_units."""
        result = batched(model)(stimulus, params, batch_size=3)
        expected = model(stimulus, params)

        assert np.allclose(np.asarray(result), np.asarray(expected))

    def test_passes_kwargs(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that keyword arguments are forwarded to the wrapped function."""
        expected_dtype = "float64"
        result = batched(model)(stimulus, params, batch_size=3, dtype=expected_dtype)

        assert keras.ops.dtype(result) == expected_dtype

    def test_decorator(
        self,
        stimulus: PRFStimulus,
        params: pd.DataFrame,
        model: Gaussian2DPRFModel,
    ):
        """Test that the decorator syntax works with batch_size as a wrapper kwarg."""

        @batched
        def batched_call(stimulus: PRFStimulus, params: pd.DataFrame) -> Tensor:
            return model(stimulus, params)

        result = batched_call(stimulus, params, batch_size=3)
        expected = model(stimulus, params)

        assert np.allclose(np.asarray(result), np.asarray(expected))
