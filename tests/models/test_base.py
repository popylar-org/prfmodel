"""Test model base classes."""

import pytest
from prfmodel.exceptions import ShapeError
from prfmodel.exceptions import ShapeMismatchError
from prfmodel.models.base import BaseCanonical
from prfmodel.models.base import BasePopulationResponse
from prfmodel.models.base import BaseStimulusEncoder
from prfmodel.utils import ModelProtocol


def test_parameter_shape_error():
    """Test that ShapeError shows correct parameter name and shape in error message."""
    param_name = "param_1"
    param_shape = (1,)
    requirement = "must have at least 2 dimensions"

    with pytest.raises(ShapeError) as excinfo:
        raise ShapeError(param_name, param_shape, requirement)

    assert param_name in str(excinfo.value)
    assert str(param_shape) in str(excinfo.value)
    assert requirement in str(excinfo.value)


def test_shape_mismatch_error():
    """Test that ShapeMismatchError shows correct arg names and shapes in error message."""
    arg1_name, arg1_shape = "arg_1", (2, 1)
    arg2_name, arg2_shape = "arg_2", (1, 1)

    with pytest.raises(ShapeMismatchError) as excinfo:
        raise ShapeMismatchError(arg1_name, arg1_shape, arg2_name, arg2_shape)

    assert arg1_name in str(excinfo.value)
    assert arg2_name in str(excinfo.value)
    assert str(arg1_shape) in str(excinfo.value)
    assert str(arg2_shape) in str(excinfo.value)


class TestBaseModel:
    """Tests for BaseModel class."""

    model_class = ModelProtocol

    def test_abstract_fail(self):
        """Test that model instantiation fails."""
        with pytest.raises(TypeError):
            _ = self.model_class()


# Inherit all checks from TestBaseModel
class TestBasePopulationResponse(TestBaseModel):
    """Tests for BasePopulationResponse class."""

    model_class = BasePopulationResponse


class TestBaseStimulusEncoder(TestBaseModel):
    """Tests for BaseStimulusEncoder class."""

    model_class = BaseStimulusEncoder


class TestBaseComposite(TestBaseModel):
    """Tests for BaseComposite class."""

    model_class = BaseCanonical
