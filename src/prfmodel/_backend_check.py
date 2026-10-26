"""Check that Keras can load a backend that prfmodel supports.

Keras loads its backend when it is first imported. If that backend is not installed, the import fails deep inside
Keras with an error that names a missing module but not the cause. This module runs before any other module of the
package imports keras and replaces those errors with ones that say how to select or install a backend.

"""

import contextlib
import difflib
import importlib.util
import json
import os
from pathlib import Path

SUPPORTED_BACKENDS = ("jax", "tensorflow", "torch")


def check_backend() -> None:
    """Check that Keras can load a backend that prfmodel supports.

    Imports keras, which loads the backend selected by the ``KERAS_BACKEND`` environment variable or, if it is not
    set, the backend in the Keras configuration file, which defaults to TensorFlow.

    Raises
    ------
    ImportError
        If the selected backend is not one that prfmodel supports or is not installed.

    """
    backend, selection = _selected_backend()

    if backend not in SUPPORTED_BACKENDS:
        msg = f"{selection}, which prfmodel does not support."
        if matches := difflib.get_close_matches(backend.lower(), SUPPORTED_BACKENDS, n=1):
            msg += f" Did you mean '{matches[0]}'?"
        msg += f" {_how_to_fix()}"
        raise ImportError(msg)

    try:
        import keras  # noqa: PLC0415 (importing keras loads the backend, which is what this checks)
    except ImportError as err:
        # A backend that is not installed shows up as a missing module of the same name. Any other import error has
        # a different cause and is raised as is.
        if (err.name or "").partition(".")[0] != backend:
            raise
        msg = f"{selection}, which is not installed. {_how_to_fix(backend)}"
        # The traceback through Keras's modules says nothing that this message does not
        raise ImportError(msg) from None

    # Only reached if Keras resolved the backend differently from _selected_backend
    if keras.backend.backend() not in SUPPORTED_BACKENDS:
        msg = f"Keras is configured to use the '{keras.backend.backend()}' backend, which prfmodel does not support."
        msg += f" {_how_to_fix()}"
        raise ImportError(msg)


def _selected_backend() -> tuple[str, str]:
    # Resolves the backend as Keras does and describes where the choice comes from. Keras ignores KERAS_BACKEND when
    # it is empty.
    if backend := os.environ.get("KERAS_BACKEND"):
        return backend, f"KERAS_BACKEND is set to '{backend}'"

    config_path = Path(os.environ.get("KERAS_HOME", "~/.keras"), "keras.json").expanduser()
    # Keras falls back to its default if the file is missing or invalid
    with contextlib.suppress(OSError, ValueError, TypeError, KeyError):
        backend = str(json.loads(config_path.read_text())["backend"])
        return backend, f"KERAS_BACKEND is not set, so Keras uses the backend '{backend}' from {config_path}"

    return "tensorflow", "KERAS_BACKEND is not set, so Keras uses its default backend 'tensorflow'"


def _how_to_fix(missing: str | None = None) -> str:
    # find_spec looks for a package without importing it, which would take seconds for a backend
    installed = [name for name in SUPPORTED_BACKENDS if name != missing and importlib.util.find_spec(name) is not None]
    hints = []

    if installed:
        names = ", ".join(f"'{name}'" for name in installed)
        hints.append(
            f"Installed backends: {names}. Select one by setting KERAS_BACKEND before importing prfmodel, for "
            f'example with os.environ["KERAS_BACKEND"] = "{installed[0]}".',
        )

    if missing is not None:
        install = "Or install" if installed else "Install"
        hints.append(f"{install} '{missing}' with pip install \"prfmodel[{missing}]\".")
    elif not installed:
        hints.append(
            'No supported backend is installed. Install one with pip install "prfmodel[jax]", '
            '"prfmodel[tensorflow]" or "prfmodel[torch]".',
        )

    return " ".join(hints)
