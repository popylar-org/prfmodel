---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: prfmodel (3.12.3)
  language: python
  name: python3
---

# Fitting a 2D population receptive field model to fMRI data from a visual experiment

**Author**: Malte Lüken (m.luken@esciencecenter.nl)

This examples shows how to fit a population receptive field (pRF) model to blood oxygenation level-dependent (BOLD) functional magnetic resonance imaging
(fMRI) data.

A pRF model maps neural activity in a brain region of interest (ROI; e.g., V1 in the human visual cortex)
to an experimental stimulus (e.g., a bar moving through the visual field). Here, we use the visual domain as an example,
where the pRF is the part of the visual field that stimulates activity in the region of interest. Because the visual
field is two-dimensional, the pRF model also has two dimensions.

Because prfmodel uses Keras for model fitting, we need to make sure that a backend is installed before we begin.
In this example, we use the TensorFlow backend.

```{code-cell} ipython3
import os
from importlib.util import find_spec

import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)

if find_spec("tensorflow") is None:
    msg = "Could not find the tensorflow package. Please install tensorflow with 'pip install .[tensorflow]'"
    raise ImportError(msg)
```

## Loading the dataset

+++

In this example, we use BOLD responses that were recorded from a single subject with a 7 Tesla MRI scanner while
they were watching a bar moving through their visual field. We load the dataset with
{py:func}`prfmodel.examples.load_dataset` (on the first function call, the dataset will be downloaded from
[Fig Share](https://doi.org/10.6084/m9.figshare.34032573)). The dataset also contains the cortical surfaces of the subject, which we
will use to visualize our results. We choose the flat surface, which shows the entire cortex of each hemisphere at once.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-aot-visual", surface_type="flat")

print(dataset)
```

When printing the `dataset` object, we can see which fields it contains. We can visualize the flat surface mesh with
nilearn.

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf

mesh = dataset.mesh

SURF_VIEW = (90, 270)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

plot_surf(mesh, hemi="both", view=SURF_VIEW, axes=ax, figure=fig)

fig.suptitle("Flat surface");
```

## Loading the BOLD response

+++

The BOLD response was recorded in a volume of voxels with a size of 2 mm. The dataset only contains the voxels that
fall inside a (dilated) gray matter mask.

```{code-cell} ipython3
response_raw = dataset.response
response_raw.shape  # shape (num_voxels, num_frames)
```

The `response_raw` object contains the BOLD response timecourse for each voxel inside the mask. It has shape
`(num_voxels, num_frames)` where `num_voxels` is the number of voxels in the mask and `num_frames` the number of time
frames of the recording. The BOLD response timecourses have 340 time frames. They have already been converted to
percent signal change (PSC) relative to a baseline of 100, so we subtract 100 to center each timecourse around zero.

```{code-cell} ipython3
response_psc = response_raw - 100.0
```

To visualize the response on the cortical surface, we need to project it from the volume onto the surface. The volume and
the surfaces of the subject share the same coordinate system, so we can use {py:func}`nilearn.surface.vol_to_surf`
for the projection. The function samples the volume at several depths between the pial surface and the white matter
surface, so we load these surfaces as well. To map a value per voxel back into the volume, we use
{py:func}`nilearn.masking.unmask` with the gray matter mask from the dataset.

```{code-cell} ipython3
import numpy as np
from nilearn.masking import unmask
from nilearn.surface import PolyMesh, vol_to_surf

# The pial and white matter surfaces are always downloaded together with the dataset
pial_mesh = PolyMesh(left=dataset.files["pia_lh"], right=dataset.files["pia_rh"])
wm_mesh = PolyMesh(left=dataset.files["wm_lh"], right=dataset.files["wm_rh"])


def project_to_surface(voxel_values: np.ndarray) -> np.ndarray:
    """Project one value per voxel in the mask onto the vertices of both hemispheres (left first)."""
    volume = unmask(voxel_values, dataset.mask)
    return np.concatenate([
        vol_to_surf(
            volume,
            pial_mesh.parts[hemi],
            inner_mesh=wm_mesh.parts[hemi],
            mask_img=dataset.mask,  # Ignore samples outside the gray matter mask
        )
        for hemi in ("left", "right")
    ])
```

We also define a helper function to plot a statistic on the surface.

```{code-cell} ipython3
from nilearn.plotting import plot_surf_stat_map


def plot_surf_stat_map_helper(
        stat_map: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        cmap: str = "inferno",
    ) -> tuple[plt.Figure, plt.Axes]:
    """Helper function to plot a surface with a stat map."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8, 6))

    plot_surf_stat_map(
        mesh,
        stat_map,
        vmin=vmin,
        vmax=vmax,
        hemi="both",
        view=SURF_VIEW,
        cmap=cmap,
        axes=ax,
        figure=fig,
        title=title,
    )

    # Expand the 3D axes to fill the space to the left of the colorbar
    surf_axes = [a for a in fig.axes if a.name == "3d"]
    cbar_axes = [a for a in fig.axes if a.name != "3d"]
    cbar_x0 = min(a.get_position().x0 for a in cbar_axes)
    for a in surf_axes:
        a.set_position([0.0, 0.0, cbar_x0 - 0.01, 0.97])

    return fig, ax
```

We can then plot the standard deviation of each timecourse on the surface.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response_psc.std(axis=1)

plot_surf_stat_map_helper(project_to_surface(response_sd), vmax=5.0, title="Response standard deviation");
```

We can see that there is high variation in the signal in the visual areas at the occipital pole (in the center of the
plot) as we would expect for a visual stimulus.

+++

## Loading the stimulus

+++

For pRF modeling, we also need the experimental stimulus that the subject has seen during the fMRI recording. The
dataset comes with a ready-made {py:class}`prfmodel.stimuli.PRFStimulus` that has already been preprocessed for pRF
modeling.

```{code-cell} ipython3
stimulus = dataset.stimulus

print(stimulus)
```

When printing the `stimulus` object, we can see the shapes of its attributes. The `design` has one frame for each
time frame of the BOLD response and 128 cells in the y- and x-dimension. The `grid` contains the y- and x-coordinates
of each cell in the visual field, which spans approximately -4 to 4 degrees of visual angle in both dimensions.

```{code-cell} ipython3
stimulus.grid.min(axis=(0, 1)), stimulus.grid.max(axis=(0, 1))
```

We can visualize the stimulus using {py:func}`prfmodel.plotting.animate_2d_prf_stimulus`.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.plotting import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=50)  # Pause 50 ms between time frames

HTML(ani.to_html5_video())
```

We can see that the stimulus consists of a bar that moves vertically and horizontally across the screen, interrupted
by blank periods. The second half of the experiment plays the bar sequence of the first half backwards in time, so each
sweep is repeated in the opposite direction. We will use this structure later for cross-validation.

+++

## Inspecting the data

Before we model the BOLD response, we look at the timecourses and inspect the quality of the data. To do this we select a subset of voxels and plot their BOLD response over time. Note that the unit of time frames is repetition time (TR) which is
0.9 seconds for this dataset.

```{code-cell} ipython3
import plotly.io as pio
import plotly.express as px

pio.renderers.default = "notebook_connected"  # Requires internet connection to work
pio.templates.default = "simple_white"

num_frames = response_psc.shape[1]

fig = px.line(
    response_psc[::500, :].T,
    animation_frame="variable",
    range_x=(0, num_frames),
    range_y=(-5, 5),
    labels={
        "index": "Time frame (in TR)",
        "value": "BOLD response (in PSC)",
        "variable": "Voxel",
    },
    title="Voxel time courses",
)
fig.update_layout(showlegend=False, height=450)
fig.show()
```

While the timecourses are quite noisy, we can see that, for some voxels, there are regular peaks in the signal. These peaks correspond
to the bar moving through the voxel's pRF. The goal of our pRF model is to predict these peaks as closely as possible.
By comparing how similar the pRF model predictions are to the observed timecourses, we can identify voxels and areas of the brain that respond to our visual stimulus. This allows us to create a stimulus-specific pRF map of the brain.

We can get an even better overview by plotting all timecourses at once in a heatmap.

```{code-cell} ipython3
aspect_ratio = response_psc.shape[1] / response_psc.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# We use matplotlib because plotly cannot handle this many voxels
im = ax.imshow(
    response_psc,
    aspect=aspect_ratio,
    cmap="inferno",
    vmin=-2,
    vmax=5,
)

ax.set_xlabel("Time frame (in TR)")
ax.set_ylabel("Voxel index")
fig.colorbar(im, ax=ax, label="BOLD response (in PSC)");
```

Again, we can see the peaks in the BOLD response of many voxels. The location of the peaks in the timecourse differs
between voxels, reflecting the different locations of their pRFs in the visual field.

+++

## Splitting the data for cross-validation

+++

A model that fits the data it was estimated on well does not necessarily predict new data well, because it can also fit
the noise in the data (overfitting). To evaluate how well our pRF model generalizes, we use cross-validation: we fit the model on the
first half of the experiment (the training set) and evaluate its predictions on the second half (the test set).
Because the second half repeats the bar sweeps of the first half in the opposite direction, both halves cover the
same locations in the visual field but have independent noise.

We split both the stimulus design and the BOLD response at the midpoint of the time axis.

```{code-cell} ipython3
from prfmodel.stimuli import PRFStimulus

num_frames_train = num_frames // 2

# The training stimulus contains the first half of the design and the same grid
stimulus_train = PRFStimulus(
    design=stimulus.design[:num_frames_train],
    grid=stimulus.grid,
    dimension_labels=stimulus.dimension_labels,
)

response_train = response_psc[:, :num_frames_train]
response_test = response_psc[:, num_frames_train:]

print(stimulus_train)
```

Some voxels in the mask do not have valid timecourses, that is, their timecourses are constant and do not contain
any signal. We filter out all voxels that have a constant timecourse in either half of the data.

```{code-cell} ipython3
response_is_valid = (response_train.std(axis=1) > 0.0) & (response_test.std(axis=1) > 0.0)

response_train_valid = response_train[response_is_valid]
response_test_valid = response_test[response_is_valid]

response_is_valid.mean()  # Fraction of valid voxels
```

## Defining the pRF model

Now that we have our BOLD response data and stimulus in place, we can create a pRF model to *predict* a response to this stimulus.
We use the most popular pRF model that is based on the seminal paper by Dumoulin and Wandell (2008):
It assumes that the stimulus (our moving bar) elicits a response that follows a
Gaussian shape in two-dimensional visual space. This response is convolved with an impulse response
that follows the shape of the hemodynamic response in the brain. Finally, a baseline and amplitude parameter shift and scale
our predicted response to match the observed BOLD response.

The {py:class}`prfmodel.models.prf.Gaussian2DPRFModel` class performs all these steps to make a combined prediction.

**Important**: We need to add a custom impulse response model to account for the fact that each time frame in the stimulus design (and in the observed timecourses) is one TR of 0.9 seconds (the default in prfmodel is 1.0 seconds). We set the resolution
of our predicted impulse response to the TR so that the final predicted model response has the same sampling rate as the observed timecourses (see also the section [](../../important_details.md)).

```{code-cell} ipython3
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel

# Define repetition time (TR)
tr = 0.9

# Create custom impulse model. Each frame is sampled at its leading edge, so the first
# frame sits at t = 0 (where the response is zero) and no offset is needed.
impulse_model = DerivativeTwoGammaImpulse(resolution=tr)
```

We can visualize the predicted impulse response. The two-gamma parameters (`delay`, `dispersion`, `undershoot`,
`u_dispersion`, `ratio`) default to the Glover HRF parameter set
(see {py:func}`~prfmodel.impulse.defaults.default_two_gamma_impulse_glover_hrf`), so we only need to set `weight_deriv`.

```{code-cell} ipython3
# Only weight_deriv is set; the two-gamma parameters use the model's default Glover HRF values
impulse_default_params = pd.DataFrame({
    "weight_deriv": [-0.5],
})

# Predict impulse response
impulse_response = impulse_model(impulse_default_params)

fig = px.line(
    pd.DataFrame({
        "Time frame (in TR)": np.arange(impulse_response.shape[1]),
        "Impulse response": impulse_response[0]
    }),
    x="Time frame (in TR)",
    y="Impulse response",
    title="Predicted impulse response",
)
fig.update_layout(height=450)
fig.show()
```

We insert the impulse model into the {py:class}`prfmodel.models.prf.Gaussian2DPRFModel` that makes combined model predictions.

```{code-cell} ipython3
# Define pRF model with custom impulse response submodel
prf_model = Gaussian2DPRFModel(
    impulse_model=impulse_model,
)
```

We define a set of starting parameters to make a combined prediction for the training stimulus with our pRF model.

```{code-cell} ipython3
# Combine pRF starting parameters with impulse response default parameters
start_params = pd.concat([pd.DataFrame(
    {
        "mu_x": [0.0],
        "mu_y": [0.0],
        "sigma": [1.0],
        "baseline": [0.0],
        "amplitude": [1.0],
    },
), impulse_default_params], axis=1)

# Make prediction with pRF model
simulated_response = prf_model(stimulus_train, start_params)

fig = px.line(
    pd.DataFrame({
        "Time frame (in TR)": np.arange(simulated_response.shape[1]),
        "BOLD response (in PSC)": simulated_response[0]
    }),
    x="Time frame (in TR)",
    y="BOLD response (in PSC)",
    title="Predicted timecourse (training stimulus)",
)
fig.update_layout(height=450)
fig.show()
```

## Fitting the pRF model

We will fit the pRF model to the training set using two stages. We begin with a grid search to find good values for our parameters of interest (`mu_x`, `mu_y`, and `sigma`). Then, we use least squares to estimate the `baseline` and `amplitude` of
our model.

> **Note**: Typically, we would also use stochastic gradient descent (SGD) to finetune our model fits after the least-squares stage. To keep the example computationally simple, we omit this step here but encourage readers to explore this approach at the end.

+++

Let's start with the grid search by defining ranges of `mu_x`, `mu_y`, and `sigma` that we want to construct a grid
of parameter values from. For `baseline` and `amplitude`, we only provide a single value so that they will stay constant
across the entire grid. The two-gamma parameters of the impulse model are omitted entirely. Instead, the impulse model supplies
them from its default Glover HRF parameter set. However, if we wanted to override the default parameters, we could also
add ranges for them here.

```{code-cell} ipython3
param_ranges = {
    "mu_x": np.linspace(-4, 4, 20),  # Range of visual field in experiment
    "mu_y": np.linspace(-4, 4, 20),
    "sigma": np.linspace(0.05, 4.0, 20),
    # delay, dispersion, undershoot, u_dispersion, and ratio use the default Glover HRF parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}
```

For all three parameters, we defined ranges of 20 values that will be used to construct the grid. That is, the
grid search will evaluate all possible combinations of these values and return the combination that fits the observed
data best. This will result in a grid containing $20 ^ 3 = 8000$ parameter combinations. This is still a relatively small grid and we recommend specifying finer grids in practice.

Let's construct the {py:class}`prfmodel.fitters.grid.GridFitter` and perform the grid search. Note that we pass the
training stimulus and set `batch_size=20` to let the {py:class}`prfmodel.fitters.grid.GridFitter`
evaluate 20 parameter combinations at the same time (which saves us some memory). By default, the `loss` (i.e., the metric to minimize between model predictions and data) is the negative correlation, which ignores differences in baseline and amplitude between model predictions and observed data. This means the data do not need to be demeaned or converted to percent signal change first, but also that `baseline` and `amplitude` cannot be estimated by the grid search itself. We fix them here and estimate them with least squares in the next step.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter

# Create grid fitter object
grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus_train,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

# Run grid search
grid_history, grid_params = grid_fitter.fit(
    data=response_train_valid,
    parameter_values=param_ranges,
    batch_size=20,
)
```

We can print the parameters estimated by the grid search.

```{code-cell} ipython3
grid_params
```

We can see that the estimates for `mu_x`, `mu_y`, and `sigma` are one combination in our grid. In the second step,
we also optimize the `amplitude` of the pRF model together with the `baseline` using least squares. This adjusts the
scale of our model predictions to scale the observed data. We set `batch_size=200` to estimate least-squares fits
for batches of voxels sequentially and save memory.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus_train,
)

# Run least squares fit
ls_history, ls_params = ls_fitter.fit(
    data=response_train_valid,
    parameters=grid_params,
    slope_name="amplitude",  # Names of parameters to be optimized with least squares
    intercept_name="baseline",
    batch_size=200,
)
```

```{code-cell} ipython3
ls_params
```

We can see that the amplitudes are substantially lower compared to the starting value (and our initial guess) for many voxels.

+++

## Evaluating the pRF model

Now that the core pRF parameters and the auxiliary baseline and amplitude parameters are optimized, we can
compare the model predictions against the observed responses. Because we want to make predictions for all voxels in
the brain, we wrap our `prf_model` in the {py:func}`prfmodel.utils.batched` modifier function. The modifier changes the behavior of the model
to make predictions for batches of voxels sequentially. This saves us a lot of memory at the expense of minimal runtime
overhead.

For the test set, we need to be careful: the hemodynamic response to the bar at the end of the first half carries
over into the beginning of the second half. If we made predictions with the second half of the design alone, the model
would miss this carry-over. Therefore, we make predictions for the *full* stimulus and split the predicted timecourses
in the same way as the observed ones. The predictions for the first half are identical to the predictions for the
training stimulus because nothing precedes the first time frame.

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(prf_model)

# Make predictions for the full stimulus with the parameters estimated on the training set
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))

pred_response_train = pred_response[:, :num_frames_train]
pred_response_test = pred_response[:, num_frames_train:]
```

We can quantify how well the predictions align with the observed timecourses using the R-squared metric. This metric
indicates the proportion of variance in the observed data explained by our model predictions. We start by comparing
the model predictions to the observed timecourses of the training set. Because we used the training set to fit our pRF
model, we are assessing its in-sample fit.

```{code-cell} ipython3
from keras.metrics import R2Score

r2_metric = R2Score(class_aggregation=None)  # Don't aggregate score over voxels

r_squared_train = np.asarray(
    r2_metric(response_train_valid.T, pred_response_train.T)
)  # Transpose to compute score across time frames
r_squared_train.shape
```

We also compute the R-squared on the test set to assess the out-of-sample fit. Note that the out-of-sample R-squared can
be negative when the predictions are worse than a flat line at the mean of the observed timecourse.

```{code-cell} ipython3
r2_metric.reset_state()
r_squared_test = np.asarray(
    r2_metric(response_test_valid.T, pred_response_test.T)
)  # Transpose to compute score across time frames
r_squared_test.shape
```

We can look at the distribution of R-squared values across voxels for both sets.

```{code-cell} ipython3
fig = px.histogram(
    pd.DataFrame({
        "Training set (in-sample)": r_squared_train,
        "Test set (out-of-sample)": r_squared_test,
    }).melt(var_name="set", value_name="r_squared"),
    x="r_squared",
    color="set",
    barmode="overlay",
    nbins=40,
    range_x=(-0.5, 1.0),
    labels={"r_squared": "R-squared", "set": ""},
).update_layout(yaxis_title="Frequency", height=450)
fig.show()
```

We can see that many voxels have a score close to zero meaning that the pRF model does not predict the observed response well. This is expected since not the entire brain responds to our relatively simple visual stimulus.
However, a substantial amount of voxels also have higher scores, suggesting that some brain areas do respond to it.
The scores on the test set are slightly lower than on the training set, because the model partly fits the noise in the
training set, which does not generalize to the test set.

We can compare the in-sample and out-of-sample R-squared for each voxel directly.

```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

ax.hexbin(r_squared_train, r_squared_test, gridsize=50, bins="log", cmap="inferno", extent=(-0.5, 1, -0.5, 1))
ax.plot([-0.5, 1.0], [-0.5, 1.0], color="gray", linestyle="--")  # Identity line

ax.set_xlabel("R-squared (training set)")
ax.set_ylabel("R-squared (test set)")
ax.set_aspect("equal");
```

Voxels whose pRF model explains a large proportion of variance in the training set also tend to explain a large
proportion in the test set, so the pRF model generalizes well to the second half of the experiment.

Let's plot the predicted and the observed timecourses for a subsample of voxels. The dashed line marks the split
between the training and the test set.

```{code-cell} ipython3
import plotly.graph_objects as go

each_k = 500

# Sort voxels according to out-of-sample R-squared
best_voxels = np.flip(np.argsort(r_squared_test))[::each_k]

response_best_voxels = response_psc[response_is_valid][best_voxels]
pred_response_best_voxels = pred_response[best_voxels]
r_squared_test_best_voxels = r_squared_test[best_voxels]
sigma_best_voxels = ls_params["sigma"].values[best_voxels]

df_obs = pd.DataFrame(response_best_voxels.T)
df_pred = pd.DataFrame(pred_response_best_voxels.T)

df_obs["source"] = "Observed"
df_pred["source"] = "Predicted"

df = pd.concat([df_obs, df_pred], axis=0)
df["time"] = np.tile(np.arange(df_obs.shape[0]), 2)

df_melted = df.melt(id_vars=["source", "time"], var_name="voxel", value_name="response")

fig = px.line(
    df_melted,
    x="time",
    y="response",
    color="source",
    animation_frame="voxel",
    range_y=[-5, 10],
    labels={
        "time": "Time frame (in TR)",
        "response": "BOLD response (in PSC)",
        "voxel": "Voxel",
        "source": "",
    },
    title=f"Observed and predicted voxel responses (subsampled voxels)",
)

# Mark the split between the training and test set
fig.add_vline(x=num_frames_train, line_dash="dash", line_color="gray")

# Add a text trace to display per-voxel stats; this will be updated in each animation frame
fig.add_trace(go.Scatter(
    x=[num_frames / 2],
    y=[4.7],
    mode="text",
    text=[f"R-squared (test) = {r_squared_test_best_voxels[0]:.3f}, sigma = {sigma_best_voxels[0]:.3f}"],
    showlegend=False,
    hoverinfo="skip",
    textfont=dict(size=13),
))

# Append a stats text update to each animation frame's trace data
for i, frame in enumerate(fig.frames):
    r2 = r_squared_test_best_voxels[i]
    sigma = sigma_best_voxels[i]
    frame.data = list(frame.data) + [go.Scatter(
        text=[f"R-squared (test) = {r2:.3f}, sigma = {sigma:.3f}"],
    )]

fig.update_layout(showlegend=True, height=450)
fig.show()
```

We can see that the model predicts the observed timecourses for the best voxels very well, not only in the training
set but also in the test set that it has not seen during fitting.

We can also visualize the out-of-sample R-squared on the flat surface mesh. To do this, we first fill the scores into an
array with one value per voxel in the mask, leaving invalid voxels as `NaN` so that they are ignored in the projection.

```{code-cell} ipython3
def fill_valid_voxels(values: np.ndarray) -> np.ndarray:
    """Fill values for the valid voxels into an array with one value per voxel in the mask."""
    values_full = np.full((response_psc.shape[0],), fill_value=np.nan)
    values_full[response_is_valid] = values
    return values_full


r_squared_test_surf = project_to_surface(fill_valid_voxels(r_squared_test))

plot_surf_stat_map_helper(
    r_squared_test_surf, vmin=0.0, vmax=1.0, title="Out-of-sample variance explained (R-squared)"
);
```

We can see that the vertices with the highest scores are located in the visual areas of both hemispheres.

+++

## Analyzing the pRF results

To analyze and interpret the pRF parameters, we will zoom in on surface vertices with an out-of-sample R-squared > 0.3
(note that this threshold is somewhat arbitrary). Because we select vertices based on the test set, which was not used
for fitting, the selection is not biased towards vertices whose fits mostly capture noise.

```{code-cell} ipython3
# Create mask for vertices above R-squared threshold
is_above_threshold = r_squared_test_surf > 0.3

# Compute proportion of vertices above threshold
is_above_threshold.mean()
```

For these vertices, we can look at different quantities to interpret the pRFs estimated by our model.

First, we look at the pRF size indicated by `sigma` and plot it on the surface.

```{code-cell} ipython3
size_surf = project_to_surface(fill_valid_voxels(ls_params["sigma"]))

plot_surf_stat_map_helper(
    np.where(is_above_threshold, size_surf, np.nan), vmin=0.0, vmax=4.0, title="pRF size (sigma)"
);
```

We can see that vertices in the early visual pathway (e.g., V1) tend to have smaller sizes than those in the higher areas.

Besides the pRF size, we can also look at the position of the pRF relative to the center of the screen. First, we
compute the angle of the center of the pRF relative to the center (i.e., the polar angle). Because the projection onto
the surface averages the values of neighboring voxels, we project the x- and y-coordinates of the pRF centers
separately and compute the angle on the surface. Averaging angles directly would give wrong results where they wrap
around from $-\pi$ to $+\pi$.

```{code-cell} ipython3
def calc_angle(mu_x: float, mu_y: float) -> float:
    """Compute the polar angle of a pRF from the x- and y-coordinate of its center."""
    return np.angle(mu_x + mu_y * 1j)


mu_x_surf = project_to_surface(fill_valid_voxels(ls_params["mu_x"]))
mu_y_surf = project_to_surface(fill_valid_voxels(ls_params["mu_y"]))

angle_surf = calc_angle(mu_x_surf, mu_y_surf)

# The polar angle is cyclic (-pi and +pi are the same direction), so we use a cyclic colormap
plot_surf_stat_map_helper(
    np.where(is_above_threshold, angle_surf, np.nan),
    vmin=-np.pi,
    vmax=np.pi,
    title="pRF center polar angle",
    cmap="hsv",
);
```

The polar angle runs counterclockwise from the right side of the screen: an angle of 0 means that the pRF center lies
to the right of the center of the screen, an angle of $\pm\pi$ that it lies to the left. The sign of the angle
distinguishes the upper (positive) from the lower (negative) half of the screen.

The surface plot shows that vertices in the left hemisphere have angles around 0, meaning that their pRF center is on
the right side of the screen. In contrast, vertices in the right hemisphere have angles around $\pm\pi$, thus, their
pRF center is located on the left side of the screen. This is what we expect from the visual system: Each hemisphere
represents the opposite half of the visual field.

Note that the angles around $\pm\pi$ in the right hemisphere wrap around. That is, $-\pi$ and $+\pi$ point in the same
direction, so the apparent jump between them is a property of the angle definition and not a discontinuity in the
estimated pRF centers. The cyclic colormap gives both ends the same color, which keeps the transition smooth.

We can also look at the eccentricity, that is, the distance of the pRF center from the center of the screen.

```{code-cell} ipython3
def calc_eccentricity(mu_x: float, mu_y: float) -> float:
    """Compute the eccentricity of a pRF from the x- and y-coordinate of its center."""
    return np.abs(mu_x + mu_y * 1j)


eccentricity_surf = calc_eccentricity(mu_x_surf, mu_y_surf)

plot_surf_stat_map_helper(
    np.where(is_above_threshold, eccentricity_surf, np.nan), vmin=0.0, vmax=4.0, title="pRF center eccentricity"
);
```

For eccentricity, we can see segments with gradual transitions in both hemispheres that correspond to pRFs that are closer or further away from the center of the screen.

+++

## Conclusion

This example showed how to fit a two-dimensional Gaussian pRF model to empirical fMRI data collected from an experiment in the visual domain. First, we projected the BOLD response data from the volume onto the cortical surface. Second, we loaded the experimental stimulus and split the stimulus and the data in half for cross-validation. Then, we defined a pRF model and optimized its parameters on the first half using a grid search followed by least-squares to adjust for baseline and amplitude differences. Finally, we evaluated the model fit on both halves and visualized the out-of-sample fit, the estimated parameters, and derived measures on the cortical surface.

## Next Steps

The predictions by our pRF model can potentially be improved. We suggest different directions for improving the pRF model fit:

- Increasing the number of points in the parameter grid for the grid search
- Finetuning the pRF model parameters with stochastic gradient descent with {py:class}`prfmodel.fitters.sgd.SGDFitter`
- Applying preprocessing steps before fitting the pRF model (e.g., high-pass filtering)
- Optimizing the impulse model parameters in the grid search
- Building a more complex pRF model (e.g., compressive spatial summation, see Kay et al., 2013)

The cross-validation can also be extended. For example, we could swap the training and test sets, fit the model on the
second half, evaluate it on the first half, and average the out-of-sample R-squared over both folds.

+++

## References

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage, 39*(2), 647–660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. Journal of Neurophysiology, 110(2), 481–494. https://doi.org/10.1152/jn.00105.2013
