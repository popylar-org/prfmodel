"""Tests for model protocol classes."""

from abc import ABC
from abc import abstractmethod
import pandas as pd
import pytest
from prfmodel.protocols import CompositeModelProtocol
from prfmodel.protocols import ModelProtocol


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
