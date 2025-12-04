"""Tests for examples."""

from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.stimulus.prf import PRFStimulus


def test_load_2d_bar_stimulus():
    """Test that load_2d_bar_stimulus returns stimulus object."""
    stimulus = load_2d_prf_bar_stimulus()
    assert isinstance(stimulus, PRFStimulus)
