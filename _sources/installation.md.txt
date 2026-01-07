# Installation

The prfmodel package is currently only available as a development version that must be installed from GitHub.
Please note that this version is still under active development and subject to constant change.
Installing prfmodel requires Python version >= 3.10 and <= 3.12.

## Installing the development version

To clone the development version of prfmodel from the GitHub repository:

```bash
git clone git@github.com:popylar-org/prfmodel.git
cd prfmodel
```

We recommend installing prfmodel in a virtual environment to prevent conflicts with other packages,
for example using [venv](https://docs.python.org/3/library/venv.html).

On Linux/MacOS (using bash/zsh):

```bash
python -m venv my_venv # Create a virtual environment called 'my_venv'
source my_venv/bin/activate # Activate the virtual environment
```

On Windows (using cmd.exe):

```console
python -m venv my_venv # Create a virtual environment called 'my_venv'
my_venv\Scripts\activate.bat # Activate the virtual environment
```

To install the development version of prfmodel from the local clone of the repository:

```bash
python -m pip install .
```

## Installing a Keras backend

prfmodel relies on [Keras](https://keras.io/) to enable users to use different backends for model fitting.
Currently, users can choose between three backends: TensorFlow, PyTorch, and JAX.

To install prfmodel with the Tensorflow backend:

```bash
python -m pip install .[tensorflow]
```

To install the PyTorch backend:

```bash
python -m pip install .[torch]
```

To install the JAX backend:

```bash
python -m pip install .[jax]
```

## Installing dependencies for package development

For those who want to contribute to the package development, you can make an editable install of the package and
install all required additional dependencies via:

```bash
git clone git@github.com:popylar-org/prfmodel.git
cd prfmodel
pip install -e .[dev]
```
