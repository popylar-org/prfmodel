"""Tests for the check of the Keras backend."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
import keras
import pytest
from prfmodel._backend_check import SUPPORTED_BACKENDS
from prfmodel._backend_check import check_backend

CURRENT_BACKEND = keras.backend.backend()
# Any backend other than the current one, which is the only one known to be installed
OTHER_BACKEND = next(name for name in SUPPORTED_BACKENDS if name != CURRENT_BACKEND)


def import_prfmodel(
    keras_home: Path,
    backend: str | None,
    blocked: tuple[str, ...] = (),
) -> subprocess.CompletedProcess:
    """Import prfmodel in a new interpreter, because Keras loads its backend only once per process.

    Packages in `blocked` fail to import as if they were not installed.

    """
    # A separate Keras home keeps the user's configuration file out of the test
    env = {**os.environ, "KERAS_HOME": str(keras_home)}
    env.pop("KERAS_BACKEND", None)
    if backend is not None:
        env["KERAS_BACKEND"] = backend

    # A None entry in sys.modules makes importing that package fail with ModuleNotFoundError
    code = "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in blocked) + "import prfmodel\n"

    return subprocess.run(  # noqa: S603 (runs this interpreter with code from this module)
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installed_backend_passes():
    """Test that the backend the tests run with passes the check."""
    check_backend()


def test_backend_from_environment_is_logged(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch):
    """Test that the check logs the backend selected by KERAS_BACKEND."""
    monkeypatch.setenv("KERAS_BACKEND", CURRENT_BACKEND)

    with caplog.at_level(logging.INFO, logger="prfmodel"):
        check_backend()

    assert f"Using the Keras backend '{CURRENT_BACKEND}' (set by KERAS_BACKEND)." in caplog.text


def test_backend_from_keras_configuration_is_logged(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the check logs the backend selected by the Keras configuration file."""
    config_path = tmp_path / "keras.json"
    config_path.write_text(json.dumps({"backend": CURRENT_BACKEND}))
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    monkeypatch.setenv("KERAS_HOME", str(tmp_path))

    with caplog.at_level(logging.INFO, logger="prfmodel"):
        check_backend()

    assert f"Using the Keras backend '{CURRENT_BACKEND}' (set in {config_path})." in caplog.text


def test_current_backend_imports(tmp_path: Path):
    """Test that prfmodel imports when KERAS_BACKEND selects an installed backend."""
    result = import_prfmodel(tmp_path, CURRENT_BACKEND)

    assert result.returncode == 0, result.stderr


def test_misspelled_backend(tmp_path: Path):
    """Test that a misspelled backend name raises an error that suggests the right name."""
    result = import_prfmodel(tmp_path, "pytorch")

    assert "ImportError: KERAS_BACKEND is set to 'pytorch', which prfmodel does not support." in result.stderr
    assert "Did you mean 'torch'?" in result.stderr


def test_unsupported_backend(tmp_path: Path):
    """Test that a backend that Keras supports but prfmodel does not raises an error."""
    result = import_prfmodel(tmp_path, "numpy")

    assert "ImportError: KERAS_BACKEND is set to 'numpy', which prfmodel does not support." in result.stderr
    assert "Did you mean" not in result.stderr


@pytest.mark.parametrize("backend", SUPPORTED_BACKENDS)
def test_backend_not_installed(tmp_path: Path, backend: str):
    """Test that selecting a backend that is not installed raises an error that says how to install it."""
    result = import_prfmodel(tmp_path, backend, blocked=(backend,))

    assert f"ImportError: KERAS_BACKEND is set to '{backend}', which is not installed." in result.stderr
    assert f'pip install "prfmodel[{backend}]"' in result.stderr


def test_default_backend_not_installed(tmp_path: Path):
    """Test that the error says that Keras falls back to TensorFlow when KERAS_BACKEND is not set."""
    result = import_prfmodel(tmp_path, None, blocked=("tensorflow",))

    assert (
        "ImportError: KERAS_BACKEND is not set, so Keras uses its default backend 'tensorflow', which is not installed."
    ) in result.stderr


def test_installed_backends_are_suggested(tmp_path: Path):
    """Test that the error suggests selecting a backend that is installed."""
    result = import_prfmodel(tmp_path, OTHER_BACKEND, blocked=(OTHER_BACKEND,))

    assert "Installed backends: " in result.stderr
    assert f"'{CURRENT_BACKEND}'" in result.stderr.partition("Installed backends: ")[2]
    assert f"Or install '{OTHER_BACKEND}'" in result.stderr


def test_unsupported_backend_in_keras_configuration(tmp_path: Path):
    """Test that an unsupported backend in the Keras configuration file raises an error that names the file."""
    config_path = tmp_path / "keras.json"
    config_path.write_text(json.dumps({"backend": "numpy"}))

    result = import_prfmodel(tmp_path, None)

    assert (
        f"ImportError: KERAS_BACKEND is not set, so Keras uses the backend 'numpy' from {config_path}, which prfmodel "
        "does not support."
    ) in result.stderr


def test_other_import_errors_are_not_replaced(tmp_path: Path):
    """Test that an import error unrelated to the backend is raised as is."""
    # Keras imports rich when it is imported
    result = import_prfmodel(tmp_path, CURRENT_BACKEND, blocked=("rich",))

    assert "ModuleNotFoundError: import of rich halted" in result.stderr
    assert "KERAS_BACKEND" not in result.stderr
