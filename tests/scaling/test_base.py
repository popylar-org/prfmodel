"""Test scaling model base classes."""

import pytest
from prfmodel.scaling.base import BaseScaling


class TestBaseScaling:
    """Tests for BaseScaling class."""

    model_class = BaseScaling

    def test_abstract_fail(self):
        """Test that model instantiation fails."""
        with pytest.raises(TypeError):
            _ = self.model_class()
