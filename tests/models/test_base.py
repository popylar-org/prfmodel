"""Test model base classes."""

import pytest
from prfmodel.models.base import BaseImpulse
from prfmodel.models.base import BaseModel
from prfmodel.models.base import BasePRFResponse
from prfmodel.models.base import BaseTemporal
from prfmodel.models.base import BatchDimensionError
from prfmodel.models.base import ShapeError


def test_parameter_shape_error():
    """Test that ParameterShapeError shows correct parameter name and shape in error message."""
    param_name = "param_1"
    param_shape = 1

    with pytest.raises(ShapeError) as excinfo:
        raise ShapeError(param_name, param_shape)

    assert param_name in str(excinfo.value)
    assert str(param_shape) in str(excinfo.value)


def test_batch_dimension_error():
    """Test that BatchDimensionError shows correct arg names and shapes in error message."""
    arg_names = ("arg_1", "arg_2")
    arg_shapes = ((2, 1), (1, 1))

    with pytest.raises(BatchDimensionError) as excinfo:
        raise BatchDimensionError(arg_names, arg_shapes)

    for arg_name, arg_shape in zip(arg_names, arg_shapes, strict=False):
        assert arg_name in str(excinfo.value)
        assert str(arg_shape[0]) in str(excinfo.value)


class TestBaseModel:
    """Tests for BaseModel class."""

    model_class = BaseModel

    def test_abstract_fail(self):
        """Test that model instantiation fails."""
        with pytest.raises(TypeError):
            _ = self.model_class()


# Inherit all checks from TestBaseModel
class TestBasePRFResponse(TestBaseModel):
    """Tests for BasePRFResponse class."""

    model_class = BasePRFResponse


class TestBaseImpulse(TestBaseModel):
    """Tests for BaseImpulse class."""

    model_class = BaseImpulse


class TestBaseTemporal(TestBaseModel):
    """Tests for BaseTemporal class."""

    model_class = BaseTemporal
