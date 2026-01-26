"""Global test configurations."""

import pytest
from prfmodel.examples import load_2d_bar_stimulus


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add an option to the pytest command line parser."""
    parser.addoption("--examples", action="store_true", default=False, help="run example tests")


def pytest_configure(config: pytest.Config) -> None:
    """Add a marker to the pytest config."""
    config.addinivalue_line("markers", "examples: mark test as example test")


def pytest_collection_modifyitems(config: pytest.Config, items: dict) -> None:
    """Add a marker to pytest mark API."""
    if config.getoption("--examples"):
        return
    skip_examples = pytest.mark.skip(reason="need --examples option to run")
    for item in items:
        if "examples" in item.keywords:
            item.add_marker(skip_examples)


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
