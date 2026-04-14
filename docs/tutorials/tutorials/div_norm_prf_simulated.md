---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: .venv-prf
  language: python
  name: python3
---

# How to fit a Divisive Normalization population receptive field model to simulated data

+++

**Author**: Angel Daza (j.daza@esciencecenter.nl)

**Difficulty**: Beginner

+++

This tutorial explains how to fit a Divisive Normalization (DN) population receptive field (pRF) model to simulated data.

A pRF model maps neural activity in a region of interest in the brain (e.g., V1 in the human visual cortex)
to an experimental stimulus (e.g., a bar moving through the visual field). The DN model was first proposed to overcome the inability of linear receptive field models to capture nonlinear phenomena, such as contrast saturation and surround suppression, observed in primary visual cortex (V1).

The DN model summarizes activation and normalization pRFs as isotropic, two-dimensional Gaussians, $G_1$ and $G_2$, centered on the same position $(x0, y0)$ in visual space $(x, y)$, but with different sizes, $σ1$ and $σ2$, and different amplitudes, $a$ and $c$, respectively. Activation and normalization also have constant “baselines” that is, an activation constant $b$ and a normalization constant $d$.

+++

---
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
the visual field changes over time. It has shape `(num_frames, width, height)`, where width and height define the number of pixels at which the visual field is recorded. The `grid` attribute maps each pixel to its xy-coordinate in the visual field (i.e., the degree of visual angle).

+++

We can visualize the stimulus using `animate_2d_stimulus`.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.stimuli import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=25)  # Pause 25 ms between time frames

HTML(ani.to_html5_video())
```

## Defining the DN pRF model

Now that we defined our stimulus, we can create a DN pRF model to predict a neural response to this stimulus.
The `DivNormGaussian2DPRFModel` runs two independent Gaussian pipelines sharing the same center (`mu_x`, `mu_y`) but with
different widths (`sigma_activation` for the activation pRF, `sigma_normalization` for the normalization pRF).
Each pipeline is encoded with the stimulus and convolved with a haemodynamic impulse response. The two responses
are then combined using the divisive normalization formula:

$$p_{DN} = \frac{(a G_1 \cdot S + b)}{(c G_2 \cdot S + d)} - \frac{b}{d}$$

where $G_1$ and $G_2$ are the activation and normalization Gaussian responses, with $a$ (`amplitude_activation`) ,
$b$ (`baseline_activation`), and $c$ (`amplitude_normalization`) , $d$
(`baseline_normalization`). The $-b/d$ term ensures a zero baseline response in
the absence of a stimulus. The normalization baseline $d$ must be positive to avoid division by zero.

```{code-cell} ipython3
from prfmodel.models.prf import DivNormGaussian2DPRFModel

prf_model = DivNormGaussian2DPRFModel()
```

To simulate a neural response to our stimulus with our DN pRF model, we need to define a set of parameters.

The list of parameters that need to be set to make model predictions can be obtained from the `parameter_names` property.

```{code-cell} ipython3
prf_model.parameter_names
```

The parameters `mu_x` and `mu_y` define the center of the pRF. `sigma_activation` and `sigma_normalization` set the
widths of the activation and normalization Gaussians (a hard requirement is that `sigma_normalization` must be at
least as large as `sigma_activation`). The parameters `delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`,
and `weight_deriv` determine the haemodynamic impulse response. `amplitude_activation` scales the activation response,
`baseline_activation` sets the numerator baseline, `amplitude_normalization` scales the normalization response, and
`baseline_normalization` sets the denominator baseline. We store the parameter values in a `pandas.DataFrame`.

```{code-cell} ipython3
import pandas as pd

true_params = pd.DataFrame(
    {
        "mu_x": [-2.1],
        "mu_y": [1.45],
        "sigma_activation": [1.35],
        "sigma_normalization": [2.7],  # sigma_normalization should be >= sigma_activation
        "delay": [6.0],
        "dispersion": [0.9],
        "undershoot": [12.0],
        "u_dispersion": [0.9],
        "ratio": [0.48],
        "weight_deriv": [-0.5],
        "amplitude_activation": [5.5], # a
        "baseline_activation": [2.0], # b
        "amplitude_normalization": [0.5], # c
        "baseline_normalization": [10.0],  # d - baseline_normalization must be > 0
    },
)
```

Using the "true" parameters, we simulate a response to our stimulus by making a prediction with our pRF model.

```{code-cell} ipython3
import matplotlib.pyplot as plt

simulated_response = prf_model(stimulus, true_params)

_ = plt.plot(simulated_response[0])
```

The predicted response shows the characteristic activation pattern of the DN model for each moving bar in our stimulus.

+++

## Fitting the pRF model

We will fit the DN pRF model using a two-step approach.
- In **Step 1**, we fit a vanilla Gaussian model to locate the pRF center (`mu_x`, `mu_y`) and size (`sigma`) using a grid search and least squares to determine the `amplitude`.
- In **Step 2**, we initialize the DN model parameters from the Gaussian fit using `init_dn_from_gaussian`: `mu_x` and `mu_y` stay the same; `sigma` becomes `sigma_activation` and `amplitude` becomes `amplitude_activation`, while `sigma_normalization` is set to `sigma_activation * sigma_ratio`. We then fine-tune the whole model with SGD, constraining `baseline_normalization > 0` to avoid division by zero.

+++

### Step 1: Fit the center Gaussian model

Let's start with a grid search over `mu_x`, `mu_y`, and `sigma` using a plain
`Gaussian2DPRFModel`. This is much faster than searching over both `sigma_activation` and `sigma_normalization`
simultaneously, and gives us a good initialisation point for the DN model.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFModel
import numpy as np

# Step 1: fit a plain Gaussian model to locate the center and size of the pRF
gaussian_center_model = Gaussian2DPRFModel()

param_ranges_gaussian = {
    "mu_x": np.linspace(-3.0, 3.0, 10),
    "mu_y": np.linspace(-3.0, 3.0, 10),
    "sigma": np.linspace(0.5, 5.0, 10),
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

For all three parameters, we defined ranges of 10 values, giving $10 \times 10 \times 10 = 1000$
parameter combinations to evaluate. Let's construct the `GridFitter` and run the grid search.

```{code-cell} ipython3
from keras.losses import CosineSimilarity
from prfmodel.fitters import GridFitter

grid_fitter = GridFitter(model=gaussian_center_model, stimulus=stimulus, loss=CosineSimilarity(reduction="none"))

grid_history, grid_params = grid_fitter.fit(
    data=simulated_response,
    parameter_values=param_ranges_gaussian,
    batch_size=20,
)
```

```{code-cell} ipython3
grid_params
```

The grid search returns the best-matching combination. The estimates for `mu_x`, `mu_y`, and
`sigma` are close to the true values but constrained to the grid.

```{code-cell} ipython3
gaussian_pred_response = gaussian_center_model(stimulus, grid_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(gaussian_pred_response[0], label="Predicted (Gaussian, grid search)")

fig.legend();
```

The Gaussian fit already captures the main shape of the response. Next we use least squares
to estimate `amplitude` and `baseline`.

+++

Using least squares, we estimate the `amplitude` and `baseline` parameters of the
Gaussian model, which will seed the DN initialisation.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

ls_fitter = LeastSquaresFitter(model=gaussian_center_model, stimulus=stimulus)

ls_history, gaussian_center_params = ls_fitter.fit(
    data=simulated_response,
    parameters=grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
)
gaussian_center_params
```

```{code-cell} ipython3
true_params
```

The Gaussian least-squares fit adjusts the scale and baseline to match the simulated response.

```{code-cell} ipython3
gaussian_pred_response = gaussian_center_model(stimulus, gaussian_center_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(gaussian_pred_response[0], label="Predicted (Gaussian, least-squares)")

fig.legend();
```

```{code-cell} ipython3
from prfmodel.models.prf import DoG2DPRFModel

dog_model = DoG2DPRFModel()
```

```{code-cell} ipython3
from prfmodel.models.prf import init_dog_from_gaussian

# Convert Gaussian fit to DoG starting parameters
dog_init_params = init_dog_from_gaussian(gaussian_center_params, sigma_ratio=2.0)
dog_init_params
```

```{code-cell} ipython3
from prfmodel.adapter import Adapter
from prfmodel.adapter import ParameterConstraint
from prfmodel.fitters.sgd import SGDFitter

dog_adapter = Adapter(transforms=[
    ParameterConstraint(["amplitude_surround"], upper=0.0),
])

sgd_fitter = SGDFitter(
    model=dog_model,
    stimulus=stimulus,
    adapter=dog_adapter,
)

sgd_history, dog_params = sgd_fitter.fit(
    data=simulated_response,
    init_parameters=dog_init_params,
    fixed_parameters=["delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv"],
)
```

```{code-cell} ipython3
dog_params
```

```{code-cell} ipython3
true_params
```

```{code-cell} ipython3
dog_pred_response = dog_model(stimulus, dog_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(dog_pred_response[0], "--", label="Predicted (SGD)")

fig.legend();
```

### Step 2: Fit the DN model

We now initialize a DN model from the fitted Gaussian parameters.
`init_dn_from_gaussian` sets:
- `sigma_activation = sigma`
- `sigma_normalization = sigma_activation * sigma_ratio` (default `sigma_ratio=2.0`)
- `amplitude_activation = amplitude` (a)
- `baseline_activation = baseline` (b, from the Gaussian fit)
- `amplitude_normalization = 1.0` (c)
- `baseline_normalization = 1.0` (d)

We then run SGD on the DN model. To enforce `baseline_normalization > 0` (required to prevent division by zero)
we pass a `ParameterConstraint(lower=0.0)` adapter.

> **Note — one-shot alternative:** It would be possible to skip the Gaussian
> pre-fit and run a joint grid search over `sigma_activation` and `sigma_normalization` followed
> by SGD directly. This one-shot approach works but is slower, the `baseline_normalization > 0`
> constraint is not automatically enforced, and the larger search space increases the risk of
> converging to a poor local minimum.

```{code-cell} ipython3
from prfmodel.models.prf import init_dn_from_dog

# # Convert Gaussian fit to DN starting parameters
dn_init_params = init_dn_from_dog(dog_init_params, baseline_normalization=10)
```

```{code-cell} ipython3
dn_init_pred_response = prf_model(stimulus, dn_init_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(dn_init_pred_response[0], "--", label="Predicted (SGD)")

fig.legend();
```

```{code-cell} ipython3
from keras import ops
from keras.optimizers import Adam
from prfmodel.adapter import Adapter, ParameterTransform

# Constrain baseline_normalization > 0 during SGD to avoid division by zero
dn_adapter = Adapter(transforms=[
    ParameterTransform(["baseline_normalization"], transform_fun=ops.log, inverse_fun=ops.exp),
])

sgd_fitter = SGDFitter(
    model=prf_model,
    stimulus=stimulus,
    adapter=dn_adapter,
    optimizer=Adam(learning_rate=0.01),
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=simulated_response,
    init_parameters=dn_init_params,
    fixed_parameters=[
        "mu_x", "mu_y", "delay", "dispersion", "undershoot", "u_dispersion", "ratio", "weight_deriv",
        "sigma_activation",
        "sigma_normalization",
        # "amplitude_activation",
        # "baseline_activation",
        # "amplitude_normalization",
        # "baseline_normalization",
    ],
)
```

```{code-cell} ipython3
sgd_params
```

```{code-cell} ipython3
true_params
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

In this tutorial, we showed how to fit a Divisive Normalization (DN) pRF model to simulated data using a
two-step workflow.

In **Step 1**, we fitted a plain Gaussian model with a grid search and least squares to efficiently locate the pRF center and size.

In **Step 2**, we used `init_dn_from_gaussian` to seed the DN model from the Gaussian fit. We set the normalization size to twice the activation size and initialized the normalization parameters at their defaults (`baseline_activation` from Gaussian fit, `amplitude_normalization=1`, `baseline_normalization=1`).

+++

## References

+++

Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118. https://doi.org/10.1073/pnas.2108713118
