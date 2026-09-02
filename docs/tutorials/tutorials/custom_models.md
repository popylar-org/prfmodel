---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: prfmodel (3.12.3.final.0)
  language: python
  name: python3
---

# Creating a custom model

+++

**Author**: Malte Lüken (m.luken@esciencecenter.nl)

**Difficulty**: Intermediate

+++

This tutorial explains how to create a custom model with prfmodel.

## Part 1: Implementing a 1D Gaussian pRF model

In the first part, I show how to implement a 1-dimensional Gaussian population receptive field (pRF) model analogous
to the existing 2-dimensional model. The 1D model is often used to model neural responses to auditory or numerosity
stimuli that lie on a single dimension (i.e., tone frequency or displayed number of objects).

+++

Because prfmodel uses Keras for model fitting, we need to make sure that a backend is installed before we begin.
In this tutorial, we use the TensorFlow backend.

```{code-cell} ipython3
import os
from importlib.util import find_spec

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

if find_spec("tensorflow") is None:
    msg = "Could not find the tensorflow package. Please install tensorflow with 'pip install .[tensorflow]'"
    raise ImportError(msg)
```

### Loading a 1D stimulus

We start by loading an example 1D {py:class}`~prfmodel.stimuli.PRFStimulus` from a numerosity experiment (for details, see {py:func}`~prfmodel.examples.load_1d_prf_lognumerosity_stimulus`).

```{code-cell} ipython3
from prfmodel.examples import load_1d_prf_lognumerosity_stimulus

stimulus = load_1d_prf_lognumerosity_stimulus()
print(stimulus)
```

We can visualize the design matrix with the displayed numerosity at each time frame on the natural and log scale.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

unique_log_numerosities = stimulus.grid[:, 0]
unique_numerosities = np.round(np.exp(unique_log_numerosities))

fig, ax = plt.subplots()

ax.imshow(stimulus.design.T, aspect=stimulus.design.shape[0]/stimulus.design.shape[1])
ax.set_xlabel("Time frame")
ax.set_ylabel("Numerosity (natural scale)")
ax.set_yticks(np.arange(len(unique_numerosities)))
ax.set_yticklabels(unique_numerosities)

secax = ax.secondary_yaxis("right")
secax.set_ylabel("Numerosity (log scale)")
secax.set_yticks(np.arange(len(unique_numerosities)))
secax.set_yticklabels(np.round(unique_log_numerosities, 2));
```

### Implementing the custom response model

Now we implement the 1D Gaussian response class by subclassing
{py:class}`~prfmodel.models.base.BasePopulationResponse` (note that the 1D Gaussian pRF response is already included in the package as {py:class}`~prfmodel.models.prf.Gaussian1DPRFResponse`). We first take a look at the docstring of the class:

```{code-cell} ipython3
from prfmodel.models.base import BasePopulationResponse

help(BasePopulationResponse)
```

We can see that `BasePopulationResponse` has two abstract methods that must be overridden when subclassing:

```
__abstractmethods__ = frozenset({'call', 'parameter_names'}).
```

1. The `parameter_names` property, which lists the parameter names the model expects.
2. The `call` method, which computes the pRF response for a given stimulus and parameter set. This method can implement an arbitrary response function, but here we re-use {py:fun}`~prfmodel.models.prf.predict_gaussian_response` from the Gaussian module, which is dimension-agnostic and works for any number of spatial dimensions.

Note that we implement `call`, not `__call__`. Model classes have two entry points:

- `__call__` is the user-facing **public facade**. It takes the types users typically work with, e.g., a
  {py:class}`~prfmodel.stimuli.PRFStimulus` holding NumPy arrays and a {py:class}`pandas.DataFrame` of
  parameters. It checks that every required parameter is present, resolves the `dtype`, converts everything to
  backend tensors, calls `call`, and converts the result back to a {py:class}`numpy.ndarray`. It is implemented
  once on the base class, and you should not override it.
- `call` is the **tensor-only kernel**. It receives a {py:class}`~prfmodel.stimuli.PRFStimulusTensors` (that only holds tensors)
  and a {py:class}`~prfmodel.utils.TensorFrame` (the parameters as tensors,
  which you select by column name exactly like a data frame), and it returns a backend tensor. Because
  everything arriving here is already a
  tensor and nothing needs validating, this is the method the fitters wrap in `tf.function`, `torch.compile`, or `jax.jit` to
  run the optimization in graph mode.

So the NumPy arrays go in and NumPy arrays come out: a prediction from `__call__` can go straight into
{py:mod}`matplotlib` or {py:mod}`scipy` with no conversion, and it works the same on GPU for every backend.
Backend tensors stay an implementation detail below `call`. When you do need a tensor, for example inside your
own `call`, call `call` directly rather than `__call__`.

That division has one rule you have to respect when writing `call`: it must be **traceable**. Use
{py:mod}`keras.ops` only, never NumPy or pandas, and never write an `if` statement that branches on a tensor *value*.
While a graph is being built, a tensor holds no value to branch on, so a check like `if ops.all(sigma > 0)`
raises an error. Branching on a tensor *shape* is fine because shapes are known while tracing. If you need to reject bad
parameter values, do it where the values are still concrete, for example by overriding the {py:meth}`check_parameter_values`
method of the model class (more on that soon).

We can also see that `BasePopulationResponse` is a generic class with respect to the stimulus.
This means we need to specify for which stimulus type the class is defined, and which tensor type matches it.
In our case, these are {py:class}`~prfmodel.stimuli.PRFStimulus` and
{py:class}`~prfmodel.stimuli.PRFStimulusTensors` (for a connective field response model, these would be
{py:class}`~prfmodel.stimuli.CFStimulus` and {py:class}`~prfmodel.stimuli.CFStimulusTensors`).

```{code-cell} ipython3
from prfmodel.models.prf import predict_gaussian_response
from prfmodel.stimuli import PRFStimulus, PRFStimulusTensors
from prfmodel.utils import TensorFrame

# Define the generic class for the concrete 'PRFStimulus' type and its matching tensor type
class Gaussian1DPRFResponse(BasePopulationResponse[PRFStimulus, PRFStimulusTensors]):
    # 'parameter_names' is a property so that it becomes "immutable"
    @property
    def parameter_names(self) -> list[str]:
        """Names of parameters used by the model: `mu`, `sigma`."""
        return ["mu", "sigma"]

    def call(
            self,
            # The 'stimulus' argument must be the tensor type from the concrete types above
            stimulus: PRFStimulusTensors,
            parameters: TensorFrame,
        ):
        """Predict the model response for a stimulus with a 1D grid.

        Parameters
        ----------
        stimulus : PRFStimulusTensors
            The stimulus arrays as tensors.
        parameters : TensorFrame
            Model parameters as tensors, selected by column name like a data frame.

        Returns
        -------
        Tensor
            Model predictions of shape `(num_units, num_coordinates)`.
            `num_units` is the number of rows in `parameters` and `num_coordinates` is the size of the
            stimulus grid dimension.

        """
        # No dtype handling and no tensor conversion here: '__call__' already did both, so 'stimulus.grid'
        # is a tensor and 'parameters[["mu"]]' returns one.
        mu = parameters[["mu"]]
        sigma = parameters[["sigma"]]
        # We can implement the Gaussian response from scratch
        # import math

        # grid = ops.expand_dims(stimulus.grid, 0)
        # mu = ops.expand_dims(mu, 1)
        # sigma_squared = ops.square(sigma)

        # # Gaussian response
        # resp = ops.sum(ops.square(grid - mu), axis=-1)
        # resp /= 2 * sigma_squared

        # # Divide by volume to normalize
        # volume = (2 * math.pi * sigma_squared) ** (1 / 2)

        # return ops.exp(-resp) / volume

        # Or we can use an existing function to predict a Gaussian response
        return predict_gaussian_response(stimulus.grid, mu, sigma)
```

The `mu` parameter defines the preferred location on the stimulus dimension (here: preferred log numerosity) and `sigma` defines the tuning width. Selecting `parameters[["mu"]]` and `parameters[["sigma"]]` gives tensors with shapes `(num_units, 1)`.

Even though we implemented `call`, we still *use* the model by calling it normally because `model(stimulus, parameters)` goes through the facade, which validates the parameters and converts them before reaching our `call`.

`predict_gaussian_response` expects `mu` and `sigma` to have at least two dimensions: the first for the number of units and the second for the number of spatial dimensions. The function then broadcasts these tensors against the stimulus `grid` to compute the Gaussian response for each unit.

### Creating the model

With the `Gaussian1DPRFResponse` class defined, we pass it as the `prf_model` argument to {py:class}`~prfmodel.models.prf.canonical.CanonicalPRFModel`. The canonical model handles stimulus encoding, impulse response convolution, and baseline amplitude scaling using default submodels. Note that the 1D Gaussian pRF model is already included in the package as {py:class}`~prfmodel.models.prf.Gaussian1DPRFModel`.

```{code-cell} ipython3
from prfmodel.models.prf.canonical import CanonicalPRFModel

model = CanonicalPRFModel(
    prf_model=Gaussian1DPRFResponse(),
)
```

We can inspect all parameters required by the composite model through the `parameter_names` property.

```{code-cell} ipython3
model.parameter_names
```

The parameters `mu` and `sigma` come from our custom `Gaussian1DPRFResponse`. The remaining parameters belong to the default impulse response model ({py:class}`~prfmodel.impulse.DerivativeTwoGammaImpulse`) and the scaling model ({py:class}`~prfmodel.scaling.BaselineAmplitude`).

### Simulating a neural response

Let's simulate predicted neural responses for each unique numerosity while keeping the tuning width fixed to `sigma = 1`.

```{code-cell} ipython3
import pandas as pd

num_units = len(unique_numerosities)

params_mu = pd.DataFrame(
    {
        "mu": unique_log_numerosities,  # We need to specify the location of the Gaussian in log space
        "sigma": [1.0] * num_units,  # We keep the tuning width fixed
        "weight_deriv": [0.5] * num_units,
        "baseline": [0.0] * num_units,
        "amplitude": [1.0] * num_units,
    }
)

prediction = model(stimulus, params_mu)
print(prediction.shape)
```

The output has shape `(8, num_frames)` -- one predicted time course for each unique numerosity.

+++

We can visualize the predicted response over time.

```{code-cell} ipython3
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "notebook_connected"  # Requires internet connection to work
pio.templates.default = "simple_white"

# Name the columns after the numerosities so the animation slider shows them instead of the column index
prediction_mu = pd.DataFrame(prediction.T, columns=unique_numerosities.astype(int))

fig = px.line(
    prediction_mu,
    animation_frame="variable",
    range_x=(0, stimulus.design.shape[0]),
    range_y=(-0.2, 0.6),
    labels={
        "index": "Time frame",
        "value": "Predicted neural response",
        "variable": "Numerosity (natural scale)",
    },
)
fig.update_layout(showlegend=False, height=450)
fig.show()
```

The predicted response peaks around the time frames at which the stimulus design passes through each units's preferred frequency, and decays afterwards due to the impulse response convolution. This is exactly what we would expect from a 1D Gaussian pRF model.

+++

We can also simulate and visualize predicted timecourses for different tuning widths `sigma`.

```{code-cell} ipython3
num_units = 10

params_sigma = pd.DataFrame(
    {
        "mu": np.log([3] * num_units),
        "sigma": np.linspace(0.5, 3.0, num_units),
        "weight_deriv": [0.5] * num_units,
        "baseline": [0.0] * num_units,
        "amplitude": [1.0] * num_units,
    }
)

prediction = model(stimulus, params_sigma)

# Name the columns after the tuning widths so the animation slider shows them instead of the column index
prediction_sigma = pd.DataFrame(
    prediction.T,
    columns=[f"{sigma:.2f}" for sigma in params_sigma["sigma"]],
)

fig = px.line(
    prediction_sigma,
    animation_frame="variable",
    range_x=(0, stimulus.design.shape[0]),
    labels={
        "index": "Time frame",
        "value": "Predicted neural response",
        "variable": "pRF width (sigma)",
    },
)
fig.update_layout(showlegend=False, height=450)
fig.show()
```

We can see that the tuning width determines the sharpness of the predicted response peaks.

## Part 2: TBD

This part will be added in a future version.

## Conclusion

In this tutorial, I showed how to create a custom 1D Gaussian pRF model for a fictional numerosity experiment. I first created a stimulus for the fictional experiment. Then, I created a custom pRF response model and inserted it into the default composite pRF model
that combines the pRF response with an impulse and scaling model.

+++

## References

+++

Harvey, B. M., Klein, B. P., Petridou, N., & Dumoulin, S. O. (2013). Topographic representation of numerosity in the human parietal cortex. *Science*, *341*(6150), 1123-1126. https://doi.org/10.1126/science.1239052
