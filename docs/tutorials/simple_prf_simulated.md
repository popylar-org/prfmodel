---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3.12
  language: python
  name: python312
---

# How to fit a population receptive field model to simulated data

+++

**Author**: Malte Lüken (m.luken@esciencecenter.nl)

**Difficulty**: Beginner

+++

This tutorial explains how to fit a population receptive field (pRF) model to simulated data.

A pRF model maps neural activity in a region of interest in the brain (e.g., V1 in the human visual cortex)
to an experimental stimulus (e.g., a bar moving through the visual field). Here, we use the visual domain as an example,
where the part of the visual field that stimulates activity in the region of interest is the pRF.

+++

Because prfmodel uses Keras for model fitting, we need to make sure that a backend is installed before we begin.
In this tutorial, we use the TensorFlow backend.

```{code-cell} ipython3
import os
from importlib.util import find_spec

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Hide tensorflow info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

if find_spec("tensorflow") is None:
    msg = "Could not find the tensorflow package. Please install tensorflow with 'pip install .[tensorflow]'"
    raise ImportError(msg)
```

## Defining the stimulus

Let's start with the first step: Defining the stimulus. In practice, we recommend that users save the stimulus they use in an experiment to a file and load it to avoid mismatches between experiment and analysis.
Because we use simulated data in this tutorial, we load an example stimulus that is included in the package.
The stimulus simulates a bar moving in different directions through a two-dimensional visual field.

```{code-cell} ipython3
from prfmodel.examples import load_2d_prf_bar_stimulus

num_frames = 200  # Simulate 200 time frames

stimulus = load_2d_prf_bar_stimulus()
print(stimulus)
```

When printing the `stimulus` object, we can see that it has three attributes. The `design` attribute defines how
the visual field changes over time. It has shape `(num_frames, width, height)`, where width and hight define the number of pixels at which the visual field is recorded. The `grid` attribute maps each pixel to its xy-coordinate in the visual field (i.e., the degree of visual angle).

+++

We can visualize the stimulus using `animate_2d_stimulus`.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.stimuli import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=25)  # Pause 25 ms between time frames

HTML(ani.to_html5_video())
```

## Defining the pRF model

Now that we defined our stimulus, we can create a pRF model to predict a neural response to this stimulus in our
(hypothetical) region of interest (e.g., V1). We use the most popular pRF model that is based on the seminal paper
by Dumoulin and Wandell (2008): It assumes that the stimulus (our moving bar) elicits a response that follows a
Gaussian shape in two-dimensional visual space. This response is then summed and convolved with an impulse response
that follows the shape of the hemodynamic response in the brain. Finally, a baseline and amplitude parameter shift and scale
our predicted response to the simulated (or observed) neural response.

The `Gaussian2DPRFModel` class performs all these steps to make a combined prediction.

```{code-cell} ipython3
from prfmodel.models.gaussian import Gaussian2DPRFModel

prf_model = Gaussian2DPRFModel()
```

To simulate a neural response to our stimulus with our Gaussian 2D pRF model, we need to define a set of parameters.

The list of parameters that need to be set to make model predictions can be obtained from the `parameter_names` property.

```{code-cell} ipython3
prf_model.parameter_names
```

The parameters `mu_x`, `mu_y`, and `sigma` define the location and size of the predicted Gaussian pRF and are of primary interest. We simulate a pRF with its center at (-2.1, 1.45) and a size of 1.35. The parameters `delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`, and `weight_deriv` determine the impulse response that is convolved with our pRF response. The parameters `baseline` and `amplitude` shift and scale our convolved response, respectively. We store the parameter values in a `pandas.DataFrame` object.

```{code-cell} ipython3
import pandas as pd

true_params = pd.DataFrame(
    {
        "mu_x": [-2.1],
        "mu_y": [1.45],
        "sigma": [1.35],
        "delay": [6.0],
        "dispersion": [0.9],
        "undershoot": [12.0],
        "u_dispersion": [0.9],
        "ratio": [0.48],
        "weight_deriv": [-0.5],
        "baseline": [10.0],
        "amplitude": [1.2],
    },
)
```

Using the "true" parameters, we simulate a response to our stimulus by making a prediction with our pRF model.

```{code-cell} ipython3
import matplotlib.pyplot as plt

simulated_response = prf_model(stimulus, true_params)

_ = plt.plot(simulated_response[0])
```

The predicted response contains increased activation followed by decreased activation compared to the baseline activity for each moving bar in our stimulus.

+++

## Fitting the pRF model

We will fit the pRF model to our simulated data using multiple stages. We begin with a grid search to find good
starting values for our parameters of interest (`mu_x`, `mu_y`, and `sigma`). Then, we use least squares to estimate the `baseline` and `amplitude` of
our model. Finally, we use stochastic gradient descent (SGD) to finetune our model fits.

+++

Let's start with the grid search by defining ranges of `mu_x`, `mu_y`, and `sigma` that we want to construct a grid
of parameter values from. For all other parameters, we only provide a single value so that they will stay constant
across the entire grid.

```{code-cell} ipython3
import numpy as np

param_ranges = {
    "mu_x": np.linspace(-3.0, 3.0, 10),
    "mu_y": np.linspace(-3.0, 3.0, 10),
    "sigma": np.linspace(0.5, 3.0, 10),
    "delay": [6.0],
    "dispersion": [0.9],
    "undershoot": [12.0],
    "u_dispersion": [0.9],
    "ratio": [0.48],
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}
```

For all three parameters, we defined ranges of 10 values that will be used to construct the grid. That is, the
grid search will evaluate all possible combinations of these values and return the combination that fits the simulated
data best. This will result in a grid containing $10 \times 10 \times 10 = 1000$ parameter combinations.

Let's construct the `GridFitter` and perform the grid search. Note that we set `chunk_size=20` to let the `GridFitter`
evaluate 20 parameter combinations at the same time (which saves us some memory).

```{code-cell} ipython3
from prfmodel.fitters.grid import GridFitter

grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus,
)

grid_history, grid_params = grid_fitter.fit(
    data=simulated_response,
    parameter_values=param_ranges,
    chunk_size=20,
)
```

```{code-cell} ipython3
grid_params
```

We can see that the estimates for `mu_x`, `mu_y`, and `sigma` are one combination in our grid. However, because the
grid did not contain the "true" parameters we used to simulate the original response, the estimates differ from the
"true" parameters.

Using the parameter estimates resulting from the grid search we can make model predictions and compare them against
the original simulated response.

```{code-cell} ipython3
grid_pred_response = prf_model(stimulus, grid_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(grid_pred_response[0], label="Predicted (grid)")

fig.legend();
```

We can see that the predicted response follows the shape of the original (true) response but still shows some deviation
in the amplitude of the activation peaks and the baseline activation.

+++

Using least squares, we can estimate the baseline and amplitude parameters of our model.

```{code-cell} ipython3
from prfmodel.fitters.linear import LeastSquaresFitter

ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus,
)

ls_history, ls_params = ls_fitter.fit(
    data=simulated_response,
    parameters=grid_params,
    slope_name="amplitude",  # Names of parameters to be optimized with least squares
    intercept_name="baseline",
)

ls_params
```

Looking at the parameters, we can see that the model compensates the deviation in the peaks by adjusting the
`baseline` and `amplitude` parameters. We can also plot the predicted response.

```{code-cell} ipython3
ls_pred_response = prf_model(stimulus, ls_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(ls_pred_response[0], label="Predicted (least-squares)")

fig.legend();
```

To finetune our model fits, we use SGD to iteratively optimize model parameters using the gradient of a loss function
that is computed between data and model predictions. The default loss function in prfmodel is the means squared error.
As initial parameters, we use the result from the grid search and least squares fit. We fix the parameters related to
the impulse response to their initial values (which are the "true" values).

```{code-cell} ipython3
from prfmodel.fitters.sgd import SGDFitter

sgd_fitter = SGDFitter(
    model=prf_model,
    stimulus=stimulus,
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=simulated_response,
    init_parameters=ls_params,
    fixed_parameters=["delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv"],
)
```

```{code-cell} ipython3
sgd_params
```

We can plot the predicted model response and see that it matches the original simulated response almost perfectly.

```{code-cell} ipython3
sgd_pred_response = prf_model(stimulus, sgd_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(sgd_pred_response[0], "--", label="Predicted (SGD)")

fig.legend();
```

## Conclusion

In this tutorial, we showed how to setup a standard Gaussian pRF model for a two-dimensional stimulus. We demonstrated
how to fit the model to simulated data (without noise) using a multi-stage workflow: First, we used a grid search to
find good starting values, then, we estimated baseline and amplitude using least squares, and finally we finetuned the
model fit using stochastic gradient descent. At each stage, we compared the predicted model response against the
original simulated response to check how well the model fit the data.

## Stay Tuned

More tutorials on fitting models to empirical data and creating custom models are in the making.

For questions and issues, please make an issue on [GitHub](https://github.com/popylar-org/prfmodel/issues) or
contact Malte Lüken (m.luken@esciencecenter.nl).

+++

## References

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage, 39*(2), 647–660. https://doi.org/10.1016/j.neuroimage.2007.09.034
