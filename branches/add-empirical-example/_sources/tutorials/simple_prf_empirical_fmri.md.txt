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

# Fitting a 2D population receptive field model to fMRI data from a visual experiment

**Author**: Malte Lüken (m.luken@esciencecenter.nl)

This examples shows how to fit a population receptive field (pRF) model to blood oxygnenation level-dependent (BOLD) functional magnetic resonance imaging
(fMRI) data.

A pRF model maps neural activity in a brain region of interest (ROI; e.g., V1 in the human visual cortex)
to an experimental stimulus (e.g., a bar moving through the visual field). Here, we use the visual domain as an example,
where the pRF is the part of the visual field that stimulates activity in the region of interest. Because the visual
field is two-dimensional, the pRF model also has two dimensions.

Because prfmodel uses Keras for model fitting, we need to make sure that a backend is installed before we begin.
In this example, we use the TensorFlow backend.

```{code-cell} ipython3
%load_ext autoreload
%autoreload 2

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

## Loading the surface

+++

Before we start modelling, we need to make sure that we have all the requirements to visualize the BOLD response data. To visualize fMRI data, we can project it onto an anatomical surface of the brain. In this tutorial, we use a template surface based on the
surface of the Human Connectome Project (see https://doi.org/10.6084/m9.figshare.13372958). For visualization, we use the nilearn package. We download the surface and load the flat meshes for both hemispheres.

```{code-cell} ipython3
from prfmodel.examples import download_surface_mesh

# Download surface mesh and load as nilearn.surface.PolyMesh object
mesh = download_surface_mesh(dest_dir="data", type="flat")
```

We can visualize the flat surface with nilearn.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from nilearn.plotting import plot_surf

SURF_VIEW = (90, 270)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 18))

plot_surf(mesh, hemi="both", view=SURF_VIEW, axes=ax, figure=fig)

fig.suptitle("Flat surface (HCP template)");
```

## Loading the BOLD response

+++

Now that we have the flat surface, we load the raw BOLD response data from a single subject that can be represented on the surface mesh. We load data from both hemispheres of the subject by setting `hemisphere="both"`.

```{code-cell} ipython3
from prfmodel.examples import load_single_subject_fmri_data

response_raw = load_single_subject_fmri_data(dest_dir="data", hemisphere="both")
response_raw.shape  # shape (num_vertices, num_frames)
```

The BOLD response time courses have 120 time frames. We convert the raw BOLD response signal to percent signal change (PSC).

```{code-cell} ipython3
response_psc = ((response_raw.T / response_raw.mean(axis=1)).T - 1.0) * 100.0
```

The `response` object contains the BOLD response timecourse for each voxel on the surface mesh. It has shape `(num_vertices, num_frames)` where `num_vertices` is the number of vertices on the surface mesh and `num_frames` the number of time frames of the recording. The timecourses from the left and right hemishperes are concatenated (left is first). Because the data has been converted to PSC, each timecourse should have a mean of zero.

First, we define a helper function to plot a statistic on the surface.

```{code-cell} ipython3
import numpy as np
from nilearn.plotting import plot_surf_stat_map

def plot_surf_stat_map_helper(
        stat_map: np.ndarray,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
    """Helper function to plot a surface with a stat map."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 18))

    plot_surf_stat_map(
        mesh,
        stat_map,
        vmin=vmin,
        vmax=vmax,
        hemi="both",
        view=SURF_VIEW,
        cmap="inferno",
        axes=ax,
        figure=fig,
        title=title,
    )

    return fig, ax
```

We can then plot the standard deviation of each timecourse on the surface.

```{code-cell} ipython3
# Calculate standard deviation of each timecourse
response_sd = response_psc.std(axis=1)

plot_surf_stat_map_helper(response_sd, vmax=5.0, title="Response standard deviation");
```

We can see that there is high variation in the signal in the visual areas (e.g., V1-V3) as we would expect for a visual stimulus.

+++

## Creating the stimulus

+++

For pRF modeling, we also need the experimental stimulus that the subject has seen during the fMRI recording. The
stimulus was created using MATLAB so, we first load the raw design matrix that is located in the `data` directory.

```{code-cell} ipython3
from scipy.io import loadmat

design_path = "data/vis_design.mat"
design = np.transpose(loadmat(design_path)["stim"])
design.shape  # (num_frames, num_x, num_y)
```

The first dimension is time which matches the time frames (second dimension) of our BOLD timecourses. The other dimensions
correspond to the x- and y-coordinates of the pixels on the screen on which the stimulus was presented. We can see that
the screen was rectangular. For pRF modelling, we prefer a square design, so we transform the design matrix into square
shape by padding the y-axis.

```{code-cell} ipython3
# Calculate difference in number of pixels between x and y dimension
pixel_offset = int((design.shape[1] - design.shape[2]) / 2)

# Pad the y-dimension with zeros to match x dimension
design_square = np.pad(
    design,
    [
        (0, 0),  # We ignore the time frame and x dimension
        (0, 0),
        (pixel_offset, pixel_offset+1),
    ],
    constant_values=0  # Pad with zeros
)
design_square.shape
```

Let's take a look at the range of values in the design.

```{code-cell} ipython3
design_square.min(), design_square.max()
```

We can see that the design still holds raw pixel color intensity values ranging from 0 to 255. For pRF modelling,
we first clip them to 0 (stimulus absent) to 1 (stimulus present). Then, we apply a Gaussian filter to smooth the
transitions between stimulus absence and presence. Note that we also swap the x- and y-dimension to match the order
that prfmodel expects.

```{code-cell} ipython3
from scipy.ndimage import gaussian_filter

design_smooth = np.moveaxis(
    gaussian_filter(
        np.clip(design_square, 0, 1),  # Clip values to interval 0 and 1
        sigma=10,
        axes=(1, 2)  # Apply filter to each time frame separately
    ),
    1, 2  # Swap x and y dimension because prfmodel expects y dimension to be first!
)
design_smooth.min(), design_smooth.max()
```

Now, the design values range between 0 and 1. However, it still has 500 pixels in each screen dimension.
This resolution is unecessarily high for our pRF model, so we only select every 5th pixel to get a lower resolution.

```{code-cell} ipython3
design_smooth_low_res = design_smooth[:, ::5, ::5]
design_smooth_low_res.shape
```

The pRF is defined on the visual field and not on the screen pixels.
Therefore, we also need to define the coordinates of each pixel in the visual field. The visual field of our stimulus
ranges from -10 to 10 degrees of visual angle.

```{code-cell} ipython3
num_grid_points = design_smooth_low_res.shape[1]

x_min, y_min, x_max, y_max = -10.0, -10.0, 10.0, 10.0

coordinates = np.stack(
    # Get all combinations of x- and y-coordinates
    np.meshgrid(
        np.linspace(x_min, x_max, num_grid_points),
        np.linspace(y_min, y_max, num_grid_points),
    ),
    axis=-1,
)
coordinates.shape
```

The `coordinates` object contains the x- and y-coordinate for every screen pixel in the visual field. We combine the
final design and the coordinates in a `PRFStimulus` container that we will later use as input for our pRF model.

```{code-cell} ipython3
from prfmodel.stimuli import PRFStimulus

stimulus = PRFStimulus(design_smooth_low_res, coordinates, ["y", "x"])  # y comes before x!

print(stimulus)
```

When printing the `stimulus` object, we can see the shapes of its attributes.

We can visualize the stimulus using `animate_2d_stimulus`.

```{code-cell} ipython3
from IPython.display import HTML
from prfmodel.stimuli import animate_2d_prf_stimulus

ani = animate_2d_prf_stimulus(stimulus, interval=50)  # Pause 50 ms between time frames

HTML(ani.to_html5_video())
```

We can see that the stimulus consists of four rectangular bars that move vertically and horizontally across the screen.

+++

## Inspecting the data

Before we model the BOLD response, we look at the timecourses and inspect the quality of the data. To do this we draw a sample of random vertices and plot their BOLD response over time. Note that the unit of time frames is repetition time (TR) which is
1.5 seconds for this dataset.

```{code-cell} ipython3
import numpy as np

num_random_vertices = 100

# Set random seed for reproducibility
rng = np.random.default_rng(123)

# Select random vertices
random_vertices = rng.choice(np.arange(response_psc.shape[0]), num_random_vertices)
response_random_vertices = response_psc[random_vertices]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(response_random_vertices.T)

ax.set_xlabel("Time frame (in TR)")
ax.set_ylabel("BOLD response (in PSC)");
```

While the timecourses are quite noisy, we can see that, for some vertices, there four regular peaks in the signal. These peaks correspond
to the bars moving through the visual field. The goal of our pRF model is to predict these peaks as closely as possible.
By comparing how similar the pRF model predictions are to the observed timecourses, we can identify vertices and areas of the brain that respond
to our visual stimulus. This allows us to create a stimulus-specific pRF map of the brain.

We can get an even better overview by plotting all timecourses at once in a heatmap.

```{code-cell} ipython3
aspect_ratio = response_psc.shape[1] / response_psc.shape[0]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

im = ax.imshow(
    response_psc,
    aspect=aspect_ratio,
    cmap="inferno",
    vmin=-2,
    vmax=5,
)

ax.set_xlabel("Time frame (in TR)")
ax.set_ylabel("Vertex index")
fig.colorbar(im, ax=ax, label="BOLD response (in PSC)");
```

Again, we can see the four peaks in the BOLD response of some vertices in both hemispheres. However, the location of the peaks in the timecourse differs between vertices and hemispheres.

+++

## Defining the pRF model

Now that we have our BOLD response data and stimulus in place, we can create a pRF model to *predict* a response to this stimulus.
We use the most popular pRF model that is based on the seminal paper by Dumoulin and Wandell (2008):
It assumes that the stimulus (our moving bar) elicits a response that follows a
Gaussian shape in two-dimensional visual space. This response is convolved with an impulse response
that follows the shape of the hemodynamic response in the brain. Finally, a baseline and amplitude parameter shift and scale
our predicted response to match the observed BOLD response.

The `Gaussian2DPRFModel` class performs all these steps to make a combined prediction. However, we need to add a custom impulse
response model to account for the fact that each time frame is one TR (1.5 seconds; the default in prfmodel is 1.0 seconds). Thus, we set the resolution
of our predicted impulse response to the TR.

```{code-cell} ipython3
from prfmodel.models.gaussian import Gaussian2DPRFModel
from prfmodel.models.impulse import DerivativeTwoGammaImpulse

# Define repetition time (TR)
tr = 1.5

# Create custom impulse response model
impulse_model = DerivativeTwoGammaImpulse(
    resolution=tr,
    offset=tr / 2.0,
)
```

We can visualize the predicted impulse response for a set of default parameters.

```{code-cell} ipython3
import pandas as pd

# Define default parameters for impulse response
impulse_default_params = pd.DataFrame({
    "delay": [6.0],
    "dispersion": [0.9],
    "undershoot": [12.0],
    "u_dispersion": [0.9],
    "ratio": [0.48],
    "weight_deriv": [-0.5],
})

# Predict impulse response with default parameters
impulse_response = np.asarray(impulse_model(impulse_default_params))

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(impulse_response.T)

ax.set_xlabel("Time frame (in TR)")
ax.set_ylabel("Impulse response");
```

We insert the impulse model into the `Gaussian2DPRFModel` that makes combined model predictions.

```{code-cell} ipython3
# Define pRF model with custom impulse response submodel
prf_model = Gaussian2DPRFModel(
    impulse_model=impulse_model,
)
```

We define a set of starting parameters to make a combined prediction with our pRF model.

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
simulated_response = np.asarray(prf_model(stimulus, start_params))

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.plot(simulated_response.T)
ax2.plot(response_random_vertices.T)

ax1.set_xlabel("Time frame (in TR)")
ax1.set_ylabel("BOLD response (in PSC)")
ax1.set_title("Predicted")
ax2.set_xlabel("Time frame (in TR)")
ax2.set_title("Observed (random vertices)");
```

We can see the expected four peaks corresponding to the moving bars. When comparing the predicted response with the observed BOLD responses, we can see that their scales differ by roughly a factor of 10. We can control the scale of the predicted response with the `amplitude` parameter.

```{code-cell} ipython3
# Adjust amplitude of starting parameters
start_params["amplitude"] = 0.1

# Make a new prediction with pRF model
simulated_response_scaled = np.asarray(prf_model(stimulus, start_params))

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))

ax1.plot(simulated_response_scaled.T)
ax2.plot(response_random_vertices.T)

ax1.set_xlabel("Time frame (in TR)")
ax1.set_ylabel("BOLD response (in PSC)")
ax1.set_title("Predicted")
ax2.set_xlabel("Time frame (in TR)")
ax2.set_title("Observed (random vertices)");
```

Now, the scale of the predicted response looks much closer to the scale of the observed responses.

Some vertices do not have valid timecourses, that is, they contain `NaN` values. We filter out all vertices that have
missing BOLD response measurements.

```{code-cell} ipython3
response_is_valid = np.all(np.isfinite(response_psc), axis=1)
response_valid = response_psc[response_is_valid]
```

## Fitting the pRF model

We will fit the pRF model to our BOLD response data using two stages. We begin with a grid search to find good values for our parameters of interest (`mu_x`, `mu_y`, and `sigma`). Then, we use least squares to estimate the `baseline` and `amplitude` of
our model.

**Note**: Typically, we would also use stochastic gradient descent (SGD) to finetune our model fits after the least-squares stage. To keep the example computationally simple, we omit this step here but encourage readers to explore this approach at the end.

+++

Let's start with the grid search by defining ranges of `mu_x`, `mu_y`, and `sigma` that we want to construct a grid
of parameter values from. For `baseline` and `amplitude`, we only provide a single value so that they will stay constant
across the entire grid. The parameters of the HRF model also remain constant at their default value.

```{code-cell} ipython3
param_ranges = {
    "mu_x": np.linspace(-10, 10, 20),  # Range of visual field in experiment
    "mu_y": np.linspace(-10, 10, 20),
    "sigma": np.linspace(0.005, 10.0, 20),
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

For all three parameters, we defined ranges of 20 values that will be used to construct the grid. That is, the
grid search will evaluate all possible combinations of these values and return the combination that fits the simulated
data best. This will result in a grid containing $20 ^ 3 = 8000$ parameter combinations. This is still a relatively small grid and we recommend specifying finer grids in practice.

Let's construct the `GridFitter` and perform the grid search. Note that we set `batch_size=20` to let the `GridFitter`
evaluate 20 parameter combinations at the same time (which saves us some memory). As the `loss` (i.e., the metric to minimize between model predictions and data), we use cosine similarity which ignores differences in scale between model predictions and observed data.

```{code-cell} ipython3
from keras.losses import CosineSimilarity
from prfmodel.fitters.grid import GridFitter

# Create grid fitter object
grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus,
    loss=CosineSimilarity(reduction="none"),  # Grid fitter needs no aggregation of loss
)

# Run grid search
grid_history, grid_params = grid_fitter.fit(
    data=response_valid,
    parameter_values=param_ranges,
    batch_size=20,
)

grid_params
```

We can see that the estimates for `mu_x`, `mu_y`, and `sigma` are one combination in our grid. In the second step,
we also optimize the `amplitude` of the pRF model together with the `baseline` using least squares. This adjusts the
scale of our model predictions to scale the observed data.

```{code-cell} ipython3
from prfmodel.fitters.linear import LeastSquaresFitter

# Create least-squares fitter
ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus,
)

# Run least squares fit
ls_history, ls_params = ls_fitter.fit(
    data=response_valid,
    parameters=grid_params,
    slope_name="amplitude",  # Names of parameters to be optimized with least squares
    intercept_name="baseline",
    batch_size=200,
)

ls_params
```

We can see that the amplitudes are substantially lower compared to the starting value (and our initial guess) for many vertices.

Now that the core pRF parameters and the auxiliary baseline and amplitude parameters are optimized, we can
compare the model predictions against the observed responses. Because we want to make predictions for all vertices in
the brain, we wrap our `prf_model` in the `batched` modifier function. The modifier changes the behavior of the model
to make predictions for batches of vertices sequentially. This saves us a lot of memory at the expense of minimal runtime
overhead.

```{code-cell} ipython3
from prfmodel.utils import batched

predict_batched = batched(prf_model)

# Make predictions with optimized parameters
pred_response = np.asarray(predict_batched(stimulus, ls_params, batch_size=200))
```

We can quantify how well the predictions align with the observed timecourses using the R-squared metric. This metric indicates the proportion of variance in the observed data explained by our model predictions.

```{code-cell} ipython3
from keras.metrics import R2Score

r2_metric = R2Score(class_aggregation=None)  # Don't aggregate score over vertices

r_squared = np.asarray(r2_metric(response_valid.T, pred_response.T))  # Transpose to compute score across time frames
r_squared.shape
```

We can look at the distribution of R-squared values across vertices.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(r_squared)
ax.set_xlabel("R-squared")
ax.set_ylabel("Frequency");
```

We can see that many vertices have a score close to zero meaning that the pRF model does not predict the observed response well. This is expected since not the entire brain responds to our relatively simple visual stimulus.
However, a substantial amount of vertices also have higher scores, suggesting that some brain areas do respond to it.

Let's plot the predicted and the observed timecourses for the vertices with the closest match.

```{code-cell} ipython3
# Only plot first 10 vertices
num_vertices = 10

best_vertices = np.flip(np.argsort(r_squared))[:num_vertices]

fig, axarr = plt.subplots(5, 2, figsize=(12, 12))

for i, ax in enumerate(axarr.flatten()):
    ax.plot(response_valid[best_vertices][i], "--", label="Observed")
    ax.plot(pred_response[best_vertices][i], label="Predicted")

axarr[0, 0].legend()

for ax in axarr[4, :]:
    ax.set_xlabel("Time frame (in TR)")

axarr[2, 0].set_ylabel("BOLD response (in PSC)")

fig.tight_layout()
```

We can see that the model predicts the observed timecourses for these vertices very well.

We can also visualize the R-squared score on the flat surface mesh.

```{code-cell} ipython3
r_squared_full = np.full((response_psc.shape[0],), fill_value=np.nan)
r_squared_full[response_is_valid] = r_squared

plot_surf_stat_map_helper(r_squared_full, vmin=0.0, vmax=1.0, title="Variance explained (R-squared)");
```

We can see that the vertices with the highest scores are located in the visual areas of both hemispheres.

+++

## Analyzing the pRF results

To analyze and interpret the pRF parameters, we will zoom in on vertices with R-squared > 0.6 which is roughly 7% of the vertices in the brain (note that this threshold is somewhat arbitrary).

```{code-cell} ipython3
# Create mask for vertices above R-squared threshold
is_above_threshold = r_squared > 0.6

# Compute proportion of vertices above threshold
is_above_threshold.mean()
```

For these vertices, we can look at different quantities to interpret the pRFs estimated by our model.

First, we look at the pRF size indicated by `sigma` and plot it on the surface.

```{code-cell} ipython3
# Allocate size vector for the entire surface
size = np.full((response_psc.shape[0],), fill_value=np.nan)

# Fill valid vertices with estimated size parameters
size[response_is_valid] = np.where(is_above_threshold, ls_params["sigma"], np.nan)

plot_surf_stat_map_helper(size, vmin=0.0, vmax=10.0, title="pRF size (sigma)");
```

We can see that vertices in the early visual pathway (e.g., V1) have smaller sizes than those in the higher areas.

Besides the pRF size, we can also look at the position of the pRF relative to the center of the screen. First, we compute the angle of the center of the pRF relative to the center (i.e., the polar angle).

```{code-cell} ipython3
def calc_angle(mu_x: float, mu_y: float) -> float:
    """Compute the polar angle of a pRF from the x- and y-coordinate of its center."""
    return np.angle(mu_x + mu_y * 1j)

angle = np.full((response_psc.shape[0],), fill_value=np.nan)

angle_valid = calc_angle(ls_params["mu_x"], ls_params["mu_y"])
angle[response_is_valid] = np.where(is_above_threshold, angle_valid, np.nan)

plot_surf_stat_map_helper(angle, vmin=-np.pi, vmax=np.pi, title="pRF center polar angle");
```

The surface plot shows that vertices in the left hemisphere have a positive pRF angle, meaning that their pRF center is on the right side of the screen. In contrast, vertices in the right hemishpere have negative pRF angle, thus, their pRF center is located on the left side of the screen.

We can also look at the eccentricity, that is, the distance of the pRF center from the center of the screen.

```{code-cell} ipython3
def calc_eccentricity(mu_x: float, mu_y: float) -> float:
    """Compute the eccentricity of a pRF from the x- and y-coordinate of its center."""
    return np.abs(mu_x + mu_y * 1j)

eccentricity = np.full((response_psc.shape[0],), fill_value=np.nan)

eccentricity_valid = calc_eccentricity(ls_params["mu_x"], ls_params["mu_y"])
eccentricity[response_is_valid] = np.where(is_above_threshold, eccentricity_valid, np.nan)

plot_surf_stat_map_helper(eccentricity, vmin=0.0, vmax=10.0, title="pRF center eccentricity");
```

For eccentricity, we can see segments with gradual transitions in both hemispheres that correspond to pRFs that are closer or further away from the center of the screen.

+++

## Conclusion

This example showed how to fit a two-dimensional Gausian pRF model to empirical fMRI data collected from an experiment in the visual domain. First, we plotted the raw BOLD response data on the cortical surface. Second, we created the experimental stimulus. Then, we defined a pRF model and optimized its parameters using a grid search followed by least-squares to adjust for baseline and amplitude differences. Finally, we visualized model fit, the estimated parameters, and derived measures on the cortical surface.

## Next steps

The predictions by our pRF model can potentially be improved. We suggest different directions for improving the pRF model fit:

- Increasing the number of points in the parameter grid for the grid search
- Finetuning the pRF model parameters with stochastic gradient descent with `prfmodel.fitters.SGDFitter`
- Applying preprocessing steps before fitting the pRF model (e.g., high-pass filtering)
- Optimizing the impulse response model parameters in the grid search
- Building a more complex pRF model (e.g., compressive spatial summation, see Kay et al., 2013)

+++

## Stay Tuned

More tutorials on fitting models to empirical data and creating custom models are in the making.

For questions and issues, please make an issue on [GitHub](https://github.com/popylar-org/prfmodel/issues) or
contact Malte Lüken (m.luken@esciencecenter.nl).

+++

## References

Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex. *NeuroImage, 39*(2), 647–660. https://doi.org/10.1016/j.neuroimage.2007.09.034

Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual cortex. Journal of Neurophysiology, 110(2), 481–494. https://doi.org/10.1152/jn.00105.2013

Van Essen, D. C., Ugurbil, K., Auerbach, E., Barch, D., Behrens, T. E. J., Bucholz, R., Chang, A., Chen, L., Corbetta, M., Curtiss, S. W., Della Penna, S., Feinberg, D., Glasser, M. F., Harel, N., Heath, A. C., Larson-Prior, L., Marcus, D., Michalareas, G., Moeller, S., … WU-Minn HCP Consortium. (2012). The Human Connectome Project: A data acquisition perspective. *NeuroImage, 62*(4), 2222–2231. https://doi.org/10.1016/j.neuroimage.2012.02.018
