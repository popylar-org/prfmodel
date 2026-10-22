
[![github license badge](https://img.shields.io/github/license/popylar-org/prfmodel)](https://github.com/popylar-org/prfmodel/blob/main/LICENSE)
[![build](https://github.com/popylar-org/prfmodel/actions/workflows/build.yml/badge.svg)](https://github.com/popylar-org/prfmodel/actions/workflows/build.yml)
[![docs](https://img.shields.io/github/actions/workflow/status/popylar-org/prfmodel/.github%2Fworkflows%2Fdocumentation.yml?label=docs)](https://github.com/popylar-org/prfmodel/actions/workflows/documentation.yml)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## How to use prfmodel

A modern Python implementation for population receptive field model fitting.

## Getting started

See our online documentation on how to [get started](https://popylar-org.github.io/prfmodel/getting_started.html)
with prfmodel.

## Installation

**Requirements**: prfmodel requires Python version >= 3.10 and <= 3.13.

prfmodel is available on [PyPI](https://pypi.org/project/prfmodel/). The package relies on
[Keras](https://keras.io/) for multi-backend model fitting. At least one backend must be installed to use prfmodel.

To install prfmodel with the Tensorflow backend, run:

```console
python -m pip install "prfmodel[tensorflow]"
```

To install it with the PyTorch backend, run:

```console
python -m pip install "prfmodel[torch]"
```

To install it with the JAX backend, run:

```console
python -m pip install "prfmodel[jax]"
```

The quotes around `prfmodel[...]` are needed in some shells (e.g., zsh, the default shell on macOS), which would
otherwise interpret the square brackets.

To install the latest development version from GitHub instead, run:

```console
python -m pip install "prfmodel[tensorflow] @ git+https://github.com/popylar-org/prfmodel.git"
```

The default backend in Keras is Tensorflow, but this can be changed by setting the `KERAS_BACKEND` environment
variable, for example, at the beginning of a Python script or Jupyter notebook:

```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import prfmodel
```

**Important**: The backend must be set before importing prfmodel or keras.
See the [Keras](https://keras.io/getting_started/) documentation for details.

## Documentation

The online documentation is available at: https://popylar-org.github.io/prfmodel/.

The local documentation can be build as HTML files with:

```console
# In prfmodel directory
cd docs/
make html
```

The documentation can then be opened in the browser from `_build/html/index.html`.

## Development

Practical information on developing prfmodel can be found in [`README.dev.md`](README.dev.md). For more in-depth information
about architecture and implementation decisions, see the [Development](https://popylar-org.github.io/prfmodel/development/index.html) section in the online documentation.

## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the
[NLeSC/python-template](https://github.com/NLeSC/python-template).

The model fitting workflow in prfmodel was inspired by [braincoder](https://github.com/Gilles86/braincoder).

## Generative AI usage

Claude Code (Opus version 4.5 - 5.0) was used to partially generate and improve code in prfmodel. All improvements were
manually evaluated and approved by the authors.

## Copyright

2026, Netherlands eScience Center, Vrije Universiteit Amsterdam, Netherlands Institute for Neuroscience
