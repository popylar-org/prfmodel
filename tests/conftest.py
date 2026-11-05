"""Global test configurations."""

import matplotlib as mpl

mpl.use("Agg")

from collections.abc import Iterator
import keras
import pandas as pd
import pytest
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel
from prfmodel.stimuli import PRFStimulus

parametrize_impulse_model = pytest.mark.parametrize("model", [None, {"delay": 6.0, "dispersion": 0.9}], indirect=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add an option to the pytest command line parser."""
    parser.addoption("--examples", action="store_true", default=False, help="run example tests")
    parser.addoption("--heavy", action="store_true", default=False, help="run computation-heavy tests")


def pytest_configure(config: pytest.Config) -> None:
    """Add a marker to the pytest config."""
    config.addinivalue_line("markers", "examples: mark test as example test")
    config.addinivalue_line("markers", "heavy: mark test as computation-heavy test")
    config.addinivalue_line("markers", "float64: mark test as using float64 (runs on the CPU on Apple MPS)")


def _default_device_supports_float64() -> bool:
    # Apple MPS has no float64 support, and the PyTorch backend uses it by default when it is available
    return keras.backend.backend() != "torch" or keras.ops.convert_to_tensor(0.0).device.type != "mps"


DEFAULT_DEVICE_SUPPORTS_FLOAT64 = _default_device_supports_float64()


def _uses_float64(item: pytest.Item) -> bool:
    if item.get_closest_marker("float64") is not None:
        return True

    callspec = getattr(item, "callspec", None)

    return callspec is not None and any(
        isinstance(value, str) and value == "float64" for value in callspec.params.values()
    )


@pytest.fixture(autouse=True)
def _float64_on_cpu(request: pytest.FixtureRequest) -> Iterator[None]:
    """Run float64 tests on the CPU if the default device does not support float64.

    A test uses float64 if it is parametrized with the value `"float64"` or has the `float64` marker. Autouse
    fixtures are set up before the other function-scoped fixtures, so these create their tensors on the CPU as well.

    """
    if DEFAULT_DEVICE_SUPPORTS_FLOAT64 or not _uses_float64(request.node):
        yield
        return

    with keras.device("cpu"):
        yield


def pytest_collection_modifyitems(config: pytest.Config, items: dict) -> None:
    """Add a marker to pytest mark API."""
    # We check the marker rather than the keywords because the keywords also hold the path of a test
    if not config.getoption("--examples"):
        skip_examples = pytest.mark.skip(reason="need --examples option to run")
        for item in items:
            if item.get_closest_marker("examples") is not None:
                item.add_marker(skip_examples)

    if not config.getoption("--heavy"):
        skip_heavy = pytest.mark.skip(reason="need --heavy option to run")
        for item in items:
            if item.get_closest_marker("heavy") is not None:
                item.add_marker(skip_heavy)


class PRFStimulusSetup:
    """Test setup for pRF stimulus object."""

    start_frame: int = 25
    end_frame: int = 75

    @pytest.fixture
    def stimulus(self):
        """2D bar prF stimulus object."""
        stimulus = load_2d_prf_bar_stimulus()

        # Select subset of time frames that contain a single vertical and horizontal bar movement across screen
        design_sub = stimulus.design[self.start_frame : self.end_frame]

        # Stimulus is immutable so we need to recreated it. The frame duration matches the impulse resolution that
        # the model regression snapshots were generated with.
        return PRFStimulus(
            design=design_sub,
            grid=stimulus.grid,
            frame_duration=1.0,
            dimension_labels=stimulus.dimension_labels,
        )


class TestSetup(PRFStimulusSetup):
    """Setup parameters and objects for fitter tests."""

    @pytest.fixture
    def model(self, request: pytest.FixtureRequest):
        """Gaussian 2D pRF model instance."""
        # Only when fixture is parameterized we access the 'param' attribute
        default_parameters = request.param if hasattr(request, "param") else None

        return Gaussian2DPRFModel(
            impulse_model=DerivativeTwoGammaImpulse(
                default_parameters=default_parameters,
            ),
        )

    @pytest.fixture
    def params(self):
        """Parameters dataframe."""
        # 3 batches
        return pd.DataFrame(
            {
                "mu_x": [-1.0, 1.0, 0.0],
                "mu_y": [1.0, -1.0, 0.0],
                "sigma": [1.0, 2.0, 3.0],
                "delay": [6.0, 6.0, 6.0],
                "dispersion": [0.9, 0.9, 0.9],
                "undershoot": [12.0, 12.0, 12.0],
                "u_dispersion": [0.9, 0.9, 0.9],
                "ratio": [0.48, 0.48, 0.48],
                "weight_deriv": [0.5, 0.5, 0.5],
                "baseline": [0.1, -0.1, 0.5],
                "amplitude": [-2.0, 1.2, 0.1],
            },
        )
