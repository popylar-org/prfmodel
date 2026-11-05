# Installation

prfmodel is available on [PyPI](https://pypi.org/project/prfmodel/). Please note that prfmodel is still in beta
(version 0.x), so its interface may change between releases.
Installing prfmodel requires Python version >= 3.10 and <= 3.13.

## Creating a virtual environment

We recommend installing prfmodel in a virtual environment to prevent conflicts with other packages,
for example using [venv](https://docs.python.org/3/library/venv.html).

````````{tab-set}

``````{tab-item} Windows
```console
python -m venv my_venv # Create a virtual environment called 'my_venv'
my_venv\Scripts\activate.bat # Activate the virtual environment
```
``````

``````{tab-item} Unix (MacOS / Linux)
```bash
python -m venv my_venv  # Create a virtual environment called 'my_venv'
source my_venv/bin/activate  # Activate the virtual environment
```
``````

````````

## Installing prfmodel

prfmodel relies on [Keras](https://keras.io/) to enable users to use different backends for model fitting.
Currently, users can choose between three backends: TensorFlow, PyTorch, and JAX. If you don't know which backend to
choose, we recommend to start with TensorFlow. At least one backend must be installed to use prfmodel.

To install prfmodel with a backend:

````````{tab-set}

``````{tab-item} Tensorflow
```bash
python -m pip install "prfmodel[tensorflow]"
```
``````

``````{tab-item} PyTorch
```bash
python -m pip install "prfmodel[torch]"
```
``````

``````{tab-item} JAX
```bash
python -m pip install "prfmodel[jax]"
```
``````

````````

The quotes around `prfmodel[...]` are needed in some shells (e.g., zsh, the default shell on macOS), which would
otherwise interpret the square brackets.

The default backend in Keras is Tensorflow, but this can be changed by setting the `KERAS_BACKEND` environment
variable, for example, at the beginning of a Python script or Jupyter notebook:

```python
import os

os.environ["KERAS_BACKEND"] = "jax"

import prfmodel
```

**Important**: The backend must be set before importing prfmodel or keras.
See the [Keras](https://keras.io/getting_started/) documentation for details.

## Installing the development version

To install the latest development version of prfmodel from the GitHub repository instead (here with the TensorFlow
backend):

```bash
python -m pip install "prfmodel[tensorflow] @ git+https://github.com/popylar-org/prfmodel.git"
```

Please note that the development version is under active development and subject to constant change.

## Installing dependencies for package development

For those who want to contribute to the package development, you can make an editable install of the package and
install all required additional dependencies via:

```bash
git clone https://github.com/popylar-org/prfmodel.git
cd prfmodel
python -m pip install -e ".[dev]"
```

See the [developer documentation](https://github.com/popylar-org/prfmodel/blob/main/README.dev.md) for more
information.
