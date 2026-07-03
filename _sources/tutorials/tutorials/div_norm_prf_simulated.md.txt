---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: venv
  language: python
  name: python3
---

# How to fit a Divisive Normalization population receptive field model to simulated data

+++

**Authors**: Angel Daza (j.daza@esciencecenter.nl) & Malte Lüken (m.luken@esciencecenter.nl)

**Difficulty**: Intermediate

+++

This tutorial explains how to fit a Divisive Normalization (DN) population receptive field (pRF) model to simulated data.

A pRF model maps neural activity in a region of interest in the brain (e.g., V1 in the human visual cortex)
to an experimental stimulus (e.g., a bar moving through the visual field). The DN pRF model (Aqil et al., 2021) aims to capture multiple
neural response phenomena, such as contrast saturation and surround suppression, using DN as a canonical neural computation. It thereby attempts to overcome the limitations of previous pRF model architectures: For example, the Gaussian pRF model cannot capture nonlinear
response phenomena at all, while the Difference of Gaussian (DoG; Zuiderbaan et al., 2012) and compressive spatial summation (CSS; Kay et al., 2013) pRF models only
capture a single nonlinear response pattern.

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

stimulus = load_2d_prf_bar_stimulus()
print(stimulus)
```

When printing the `stimulus` object, we can see that it has three attributes. The `design` attribute defines how
the visual field changes over time. It has shape `(num_frames, width, height)`, where width and height define the number of pixels at which the visual field is recorded. The `grid` attribute maps each pixel to its xy-coordinate in the visual field (i.e., the degree of visual angle).

+++

We can visualize the stimulus using `animate_2d_stimulus`.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=25)  # Pause 25 ms between time frames

HTML(ani.to_html5_video())
```

## Defining the DN pRF model

Now that we defined our stimulus, we can create a DN pRF model to predict a neural response to this stimulus. We use
the DN pRF model as described in Aqil et al. (2021).
The `DivNormGaussian2DPRFModel` has two independent Gaussian pRFs sharing the same center (`mu_x`, `mu_y`) but with
different widths (`sigma_activation` for the activation pRF, `sigma_normalization` for the normalization pRF).
The two RF responses are encoded with the stimulus and combined using the DN formula:

$$p_{DN} = \frac{(a G_1 \cdot S + b)}{(c G_2 \cdot S + d)} - \frac{b}{d}$$

where $G_1$ and $G_2$ are the activation and normalization Gaussian pRF responses and S is the stimulus, with $a$ (`amplitude_activation`) ,
$b$ (`baseline_activation`), and $c$ (`amplitude_normalization`) , $d$
(`baseline_normalization`). The $-b/d$ term ensures a zero baseline response in
the absence of a stimulus. The normalization baseline $d$ must be positive to avoid division by zero. Importantly, $b$ (`baseline_activation`) indicates the amount of surround suppression, whereas $d$ (`baseline_normalization`) indicates
inverse compression (i.e., lower $d$ means more compression). Thus, by varying $b$ and $d$, the DN pRF model can capture
different combinations of surround suppression and compression.

We create an instance of the DN pRF model.

```{code-cell} ipython3
from prfmodel.models.prf import DivNormGaussian2DPRFModel

dn_model = DivNormGaussian2DPRFModel()
```

To simulate a neural response to our stimulus with our DN pRF model, we need to define a set of parameters.

The list of parameters that need to be set to make model predictions can be obtained from the `parameter_names` property.

```{code-cell} ipython3
dn_model.parameter_names
```

The parameters `mu_x` and `mu_y` define the center of the pRF. `sigma_activation` and `sigma_normalization` set the
widths of the activation and normalization Gaussians (`sigma_normalization` should be larger than `sigma_activation`). The parameters `delay`, `dispersion`, `undershoot`, `u_dispersion`, `ratio`,
and `weight_deriv` determine the impulse response. The two-gamma parameters (`delay`, `dispersion`,
`undershoot`, `u_dispersion`, `ratio`) default to the Glover HRF parameter set
(see {py:func}`~prfmodel.impulse.defaults.default_two_gamma_impulse_glover_hrf`), so we only set `weight_deriv` here.
`amplitude_activation` scales the activation RF response,
`baseline_activation` sets the numerator baseline, `amplitude_normalization` scales the normalization RF response, and
`baseline_normalization` sets the denominator baseline. We store the parameter values in a `pandas.DataFrame`.

```{code-cell} ipython3
import pandas as pd

true_params = pd.DataFrame(
    {
        "mu_x": [-2.1],
        "mu_y": [1.45],
        "sigma_activation": [1.0],
        "sigma_normalization": [4.0],  # sigma_normalization should be > sigma_activation
        # delay, dispersion, undershoot, u_dispersion, and ratio use the default Glover HRF parameters
        "weight_deriv": [-0.5],
        "amplitude_activation": [1.0], # a
        "baseline_activation": [10], # b
        "amplitude_normalization": [1.0], # c
        "baseline_normalization": [20],  # d - baseline_normalization must be > 0
        "baseline": [0.0],  # final baseline
    },
)
```

Using the "true" parameters, we simulate a response to our stimulus by making a prediction with our pRF model.

```{code-cell} ipython3
import matplotlib.pyplot as plt

simulated_response = dn_model(stimulus, true_params)

_ = plt.plot(simulated_response[0])
```

The predicted response shows both surround suppresion (e.g., see the dip before the first peak) and compression (see the sharp response peaks).

+++

## Fitting the pRF model

The DN model has four additional parameters compared to the Gaussian pRF model (i.e., it is *overparameterized*). This means that it can be challenging to fit the model to data because different parameter sets can lead to similar model predictions. To mitigate this problem, we can provide the model with good starting values, so that iterative fitting algorithms such as stochastic gradient descent (SGD) do not get stuck in local minima.

We will fit the DN pRF model using a multi-step approach.
- In **Step 1**, we fit a Gaussian model to locate the pRF center (`mu_x`, `mu_y`) and size (`sigma`) using a grid search and least squares to determine the `amplitude`.
- In **Step 2**, we use the Gaussian parameters to initialize the Difference of Gaussian (DoG) and Compressive Spatial Summation (CSS) models that we both fit to data with SGD. The parameters estimates from the DoG model provide good starting values for `baseline_activation` because they capture surround suppression. The estimates form the CSS model, in contrast, provide good starting values for `baseline_normalization` because they capture nonlinear compression effects.
- In **Step 3**, we initialize the DN model parameters from estimates from the Gaussian, DoG, and CSS model using {py:func}`~prfmodel.models.prf.init_div_norm_from_dog_css`. We also fit the DN model to data with SGD.

+++

### Step 1: Fitting the Gaussian model

We start with a grid search over `mu_x`, `mu_y`, and `sigma` using the
`Gaussian2DPRFModel`. We will use estimates for these parameters as starting values for all subsequent models.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DPRFModel
import numpy as np

# Step 1: fit a plain Gaussian model to locate the center and size of the pRF
gaussian_model = Gaussian2DPRFModel()

param_ranges_gaussian = {
    "mu_x": np.linspace(-3.0, 3.0, 10),
    "mu_y": np.linspace(-3.0, 3.0, 10),
    "sigma": np.linspace(0.5, 5.0, 20),
    # delay, dispersion, undershoot, u_dispersion, and ratio use the default Glover HRF parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}
```

For all three parameters, we defined ranges of 10 values, giving the fitter $10 \times 10 \times 10 = 1000$
parameter combinations to evaluate. Let's construct the `GridFitter` and run the grid search. Note that we are using
a cosine similarity loss function that ignores differences in scale between model predictions and data.

```{code-cell} ipython3
from keras.losses import CosineSimilarity
from prfmodel.fitters import GridFitter

grid_fitter = GridFitter(
    model=gaussian_model,
    stimulus=stimulus,
    loss=CosineSimilarity(reduction="none"),
)

grid_history, grid_params = grid_fitter.fit(
    data=simulated_response,
    parameter_values=param_ranges_gaussian,
    batch_size=20,
)

grid_params
```

Because our grid search was agnostic to scale difference between data and predicted response, we use least squares to
estimate the amplitude of our pRF model that accounts for such scale differences.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

ls_fitter = LeastSquaresFitter(
    model=gaussian_model,
    stimulus=stimulus,
)

ls_history, ls_params = ls_fitter.fit(
    data=simulated_response,
    parameters=grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
)

ls_params
```

We make a prediction with our estimated parameters and compare against the true DN response.

```{code-cell} ipython3
gaussian_pred_response = gaussian_model(stimulus, ls_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(gaussian_pred_response[0], label="Predicted (Gaussian)")

fig.legend();
```

Now we can see that the Gaussian model does not quite capture the shape of the true DN response. This is because it cannot account
for surround suppression or compression which are inherent to the DN response.

+++

### Step 2: Fitting the DoG and CSS model

To find good starting values of the DN parameters that capture surround suppresion and compression, we fit both the DoG
and the CSS model using the Gaussian parameter estimates as starting values.

We start with the DoG model using the helper function {py:func}`~prfmodel.models.prf.init_dog_from_gaussian`.

```{code-cell} ipython3
from prfmodel.models.prf import DoG2DPRFModel
from prfmodel.models.prf import init_dog_from_gaussian

dog_model = DoG2DPRFModel()

# Convert Gaussian fit to DoG starting parameters
dog_init_params = init_dog_from_gaussian(ls_params)
```

We fit the DoG model to the true DN response using SGD. We also use an adapter to optimize the strictly positive parameters
`sigma_center` and `sigma_surround` on the log-scale.

```{code-cell} ipython3
from keras import ops
from prfmodel.fitters import SGDFitter
from prfmodel.fitters.adapter import Adapter, ParameterTransform

adapter = Adapter([
    ParameterTransform(
        parameter_names=["sigma_center", "sigma_surround"],
        transform_fun=ops.log,
        inverse_fun=ops.exp
    ),
])

sgd_fitter = SGDFitter(
    model=dog_model,
    stimulus=stimulus,
    adapter=adapter,
)

sgd_history, dog_params = sgd_fitter.fit(
    data=simulated_response,
    init_parameters=dog_init_params,
    fixed_parameters=["weight_deriv"],
    num_steps=250,
)

dog_params
```

We can make a prediction with the DoG model to see how well it fits the true DN response.

```{code-cell} ipython3
dog_pred_response = dog_model(stimulus, dog_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(dog_pred_response[0], "--", label="Predicted (DoG)")

fig.legend();
```

We can see that the DoG model captures the shape of the true DN response much better than the Gaussian model. However, the fit is still not perfect because the DoG model cannot capture the compression in the true DN response.

+++

We also fit the CSS model to the true DN response to get a starting value for the compression parameter in the DN model. Here, we use an adapter
to optimize strictly positive parameters `sigma` and `n` (i.e., the compression exponent) on the log-scale.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian2DCSSPRFModel
from prfmodel.models.prf import init_css_from_gaussian

css_model = Gaussian2DCSSPRFModel()

# Convert Gaussian fit to CSS starting parameters
css_init_params = init_css_from_gaussian(ls_params)

adapter = Adapter([
    ParameterTransform(
        parameter_names=["sigma", "n"],
        transform_fun=ops.log,
        inverse_fun=ops.exp
    ),
])

# Fit the CSS model with SGD
sgd_fitter = SGDFitter(
    model=css_model,
    stimulus=stimulus,
    adapter=adapter,
)

sgd_history, css_params = sgd_fitter.fit(
    data=simulated_response,
    init_parameters=css_init_params,
    fixed_parameters=["weight_deriv", "gain"],
    num_steps=500,
)

css_params
```

We can also make a prediction with the CSS model to see how well it fits the true DN response.

```{code-cell} ipython3
css_pred_response = css_model(stimulus, css_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(css_pred_response[0], "--", label="Predicted (CSS)")

fig.legend();
```

We can see that it fits the true DN response worse than the DoG model because compression is weaker than surround suppresion.

+++

### Step 3: Fitting the DN model

Using parameter estimates from the Gaussian, DoG, and CSS model, we now initialize the DN model using {py:func}`~prfmodel.models.prf.init_div_norm_from_dog_css`. The function uses several approximations to find good starting values for the DN model parameters (a tutorial on this will follow soon; TODO).

```{code-cell} ipython3
from prfmodel.models.prf import init_div_norm_from_dog_css

dn_init_params = init_div_norm_from_dog_css(
    dog_params,
    css_n=css_params["n"],
    stimulus=stimulus,
)

pd.concat([
    true_params.loc[:, dn_init_params.columns],
    dn_init_params
]).set_index(keys=[["true", "init"]])
```

We can see that the initial DN parameters are already quite close to the true DN parameters (only `sigma_normalization` and `baseline_activation` are off).

We can use them to make a prediction with the DN model and compare against the true DN response.

```{code-cell} ipython3
dn_init_pred_response = dn_model(stimulus, dn_init_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(dn_init_pred_response[0], "--", label="Predicted (DN, init)")

fig.legend();
```

We can see that the initial parameters already capture the true DN response to some degree. They capture the surround suppression but not the compression (i.e., the sharpness of the peaks).

We use SGD to optimize the DN parameters. Again, we use an adapter to optimize the strictly positive DN parameters on the log-scale. We also use a slightly higher learning rate in the SGD optimizer.

```{code-cell} ipython3
from keras.optimizers import Adam

adapter = Adapter([
    ParameterTransform(
        [
            "sigma_activation",
            "sigma_normalization",
            "amplitude_activation",
            "amplitude_normalization",
            "baseline_activation",
            "baseline_normalization",
        ],
        ops.log,
        ops.exp
    )
])

sgd_fitter = SGDFitter(
    model=dn_model,
    stimulus=stimulus,
    optimizer=Adam(learning_rate=0.01),
    adapter=adapter,
)

sgd_history, dn_params = sgd_fitter.fit(
    data=simulated_response,
    init_parameters=dn_init_params,
    fixed_parameters=[
        "weight_deriv",
        "amplitude_normalization",
    ],
    num_steps=500,
)
dn_params
```

We can make a prediction with the estimated DN parameters and compare it against the true DN response.

```{code-cell} ipython3
dn_pred_response = dn_model(stimulus, dn_params)

fig, ax = plt.subplots()

ax.plot(simulated_response[0], label="True")
ax.plot(dn_pred_response[0], "--", label="Predicted (DN, SGD)")

fig.legend();
```

We can see that it matches the original simulated response perfectly.

We can also compare the estimated DN parameters against the true DN parameters.

```{code-cell} ipython3
pd.concat([
    true_params.loc[:, dn_params.columns],
    dn_params
]).set_index(keys=[["true", "est"]])
```

The estimated DN parameters align closely with the true DN parameters.

+++

## Conclusion

In this tutorial, we showed how to fit a Divisive Normalization (DN) pRF model to simulated data using a
multi-step workflow.

In **Step 1**, we fitted a Gaussian model with a grid search and least squares to efficiently locate the pRF center and size.

In **Step 2**, we used the Gaussian parameters estimates as starting values to fit the Difference of Gaussian (DoG) and Compressive Spatial Summation (CSS) models using Stochastic Gradient Descent (SGD).

In **Step 3**, we used {py:func}`~prfmodel.models.prf.init_div_norm_from_dog_css` to approximate good starting values for the DN model from the DoG and CSS parameter estimates. Using the starting values, we fitted the DN model with SGD and found that the resulting parameter estimates were close to the data-generating parameters and provided a perfect fit to the simulated timecourse.

+++

## References

+++

Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response signatures throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46), e2108713118. https://doi.org/10.1073/pnas.2108713118

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. *Journal of Neurophysiology, 110*(2), 481–494. https://doi.org/10.1152/jn.00105.2013

Zuiderbaan, W., Harvey, B. M., & Dumoulin, S. O. (2012). Modeling center–surround configurations in population receptive fields using fMRI. *Journal of Vision*, *12*(3), 10. https://doi.org/10.1167/12.3.10
