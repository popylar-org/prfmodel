"""Global test configurations."""

import pytest
from prfmodel.examples import load_2d_bar_stimulus


class StimulusSetup:
    """Test setup for stimulus object."""

    start_frame: int = 40
    end_frame: int = 65

    @pytest.fixture
    def stimulus(self):
        """2D bar stimulus object."""
        stimululus = load_2d_bar_stimulus()

        # Select subset of time frames that contain a single bar movement across screen
        stimululus.design = stimululus.design[self.start_frame : self.end_frame]

        return stimululus
