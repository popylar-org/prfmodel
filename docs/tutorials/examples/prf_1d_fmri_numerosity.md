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

# Fitting a 1D population receptive field model to fMRI data from a numerosity experiment

**Author**: Malte Lüken (m.luken@esciencecenter.nl) & Blazej M. Baczkowski (b.baczkowski@esciencecenter.nl)

This example shows how to fit a population receptive field (pRF) model to functional magnetic resonance imaging (fMRI) data. The data was recorded while a subject viewed displays with different numbers of dots.

A pRF model describes which part of a stimulus space drives the response at a single location in the brain. Here that location is a voxel (a three-dimensional pixel of the fMRI images), and its response -- the BOLD signal -- reflects the pooled activity of the underlying neuronal population. In the visual domain, the stimulus space is the visual field, and the pRF is a region within it (see [the example on population receptive fields](prf_2d_fmri_visual.md)). In this example, the stimulus space is numerosity: the number of items in a display (Harvey et al., 2013). We investigate whether a cortical region has its preferred numerosity, i.e., it responds most strongly to one numerosity and more weakly as a numerosity moves away from that value, which defines a tuning curve. Because numerosity is a single quantity, this pRF is one-dimensional. Following Harvey et al. (2013), we define it over the logarithm scale (_log numerosity_). This follows from the behaviour of numerosity perception, which obeys Weber’s law: the discriminability of two numerosities depends on their ratio rather than their absolute difference, so tuning width grows roughly in proportion to preferred numerosity. On a logarithmic axis this scaling is absorbed, leaving tuning curves that are more symmetric -- approximately Gaussian -- and more uniform in width across the map, which makes them well suited to a Gaussian pRF model.

The example first prepares the recorded responses for model fitting, then fits the model to the voxels in three stages (grid search, least squares, and stochastic gradient descent), and finally interprets the estimated parameters, both in the voxels and projected onto the cortical surface.

Because prfmodel uses Keras for model fitting, we need to make sure that a backend is installed before we begin.
In this example, we use the TensorFlow backend.

```{code-cell} ipython3
import os
from importlib.util import find_spec

import numpy as np
import pandas as pd

# Set keras backend to 'tensorflow' (this is normally the default)
os.environ["KERAS_BACKEND"] = "tensorflow"
# Print parameter DataFrames with three decimals
pd.set_option("display.precision", 3)

if find_spec("tensorflow") is None:
    msg = 'Could not find the tensorflow package. Please install it with: pip install "prfmodel[tensorflow]"'
    raise ImportError(msg)
```

## Loading the stimulus

In the experiment by Hendrikx et al. (2024), the subject viewed a sequence of dot displays with different numerosities.
This stimulus is included in prfmodel and can be loaded with
{py:func}`prfmodel.examples.load_1d_prf_lognumerosity_stimulus`.


```{code-cell} ipython3
from prfmodel.examples import load_1d_prf_lognumerosity_stimulus

stimulus = load_1d_prf_lognumerosity_stimulus()
print(stimulus)
```


As in the [2D example](prf_2d_fmri_visual.md), the stimulus is a {py:class}`~prfmodel.stimuli.PRFStimulus`, which describes what the subject saw at each time frame (see also the section on stimulus encoding in the [](../../introduction.md)). It has three attributes:

- `design`: What is shown at each time frame, with shape `(num_frames, num_coordinates)`. Here, each column belongs to one numerosity, and at each time frame, the column of the shown numerosity is one and all other columns are zero.
- `grid`: The coordinates of the columns in the feature space of the pRF, with shape `(num_coordinates, 1)`. Here, they are the log numerosities.
- `frame_duration`: The duration of one time frame in seconds, which is the repetition time (TR) of the recording.


The printed stimulus shows that the experiment has 182 time frames of 2.1 seconds each and eight numerosities. We plot the design with {py:func}`prfmodel.plotting.plot_1d_prf_stimulus` to see how numerosity changes over time. The left axis shows the numerosities on the natural scale and the right axis their log values from the grid.

```{code-cell} ipython3
from prfmodel.plotting import plot_1d_prf_stimulus

# Log numerosity of each column of the design and the numerosity on the natural scale
unique_log_numerosities = stimulus.grid.squeeze()
unique_numerosities = np.round(np.exp(unique_log_numerosities))

plot_1d_prf_stimulus(
    stimulus,
    tick_labels=unique_numerosities.astype(int),
    secondary_tick_labels=np.round(unique_log_numerosities, 2),
    ylabel="Numerosity (natural scale)",
    secondary_ylabel="Numerosity (log scale)",
);
```

The stimulus consists of a cycle that is repeated four times: The numerosities 1 to 7 are shown in ascending order, followed by numerosity 20, then the numerosities 7 to 1 in descending order, followed by numerosity 20 again. Each numerosity from 1 to 7 is shown for two time frames and numerosity 20 for eight time frames, so one cycle lasts 44 time frames. Numerosity 20 serves as a baseline. Before the first cycle, there is a pre-scan interval of six time frames that also shows numerosity 20, which gives $6 + 4 \times 44 = 182$ time frames in total. We store these numbers for later, because we need them to match the model predictions to the responses.

```{code-cell} ipython3
NUM_PRESCAN_FRAMES = 6
NUM_CYCLES = 4
NUM_CYCLE_FRAMES = 44

assert stimulus.design.shape[0] == NUM_PRESCAN_FRAMES + NUM_CYCLES * NUM_CYCLE_FRAMES
stimulus.design.shape  # (time frames, numerosities)

```

This design determines when a neuron population responds. A population that prefers numerosity 2 responds early in the ascending sequence and again late in the descending sequence, so its two responses per cycle lie far apart. A population that prefers numerosity 6 responds shortly before and after the baseline, so its two responses lie close together. The recorded responses follow these times with a delay of a few seconds, because the BOLD response is slow.
We will see this pattern in the model predictions below.

The baseline shows numerosity 20 rather than a blank screen. This way, dots are visible throughout the experiment, and a change in the response reflects a change in numerosity rather than the appearance of dots. Numerosity 20 is far from the numerosities that most populations prefer, so it elicits only a weak response.



+++

## Loading the BOLD response

Now that we have the stimulus, we load the recorded BOLD responses of a single subject. We use a preprocessed version of the dataset (Mauro, 2026) that has the tag `numerosity-four-conditions`. It contains the responses to four stimulus conditions. In this example, we will use the average of the four conditions, which is the `average` task of the dataset.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# The dataset is downloaded on first use and cached in a user data directory
dataset = load_dataset("numerosity-four-conditions", task="average")
print(dataset)
```

The responses are given in volume space: `dataset.response` contains one timecourse for each voxel inside the brain mask `dataset.mask`, a three-dimensional image of the brain. The rows of the response follow the order of the voxels inside the mask, so results can be put back into the volume later. In the dataset, the six time frames of the pre-scan interval were already removed and the four cycles averaged, so each timecourse has 44 time frames.

```{code-cell} ipython3
response = dataset.response

assert response.shape[1] == NUM_CYCLE_FRAMES
assert response.shape[0] == np.count_nonzero(dataset.mask.dataobj)

response.shape
```

The scan only covered part of the brain. Voxels that fell outside the scanned volume during a run have a value of zero at the affected time frames, and the dataset marks them as invalid in the `valid` column of the `units` table, which has one row per voxel. We only use the valid voxels.

```{code-cell} ipython3
is_valid = dataset.units["valid"].to_numpy()

is_valid.mean()  # Fraction of valid voxels
```

The scanner signal of a voxel has an arbitrary scale, which differs between voxels. To put all voxels on the same scale, we convert each timecourse into percent signal change (PSC): its deviation from its mean over time, in percent of that mean.

```{code-cell} ipython3
def percent_signal_change(response: np.ndarray) -> np.ndarray:
    """Express each timecourse as its deviation from its mean over time, in percent of that mean."""
    mean = response.mean(axis=1, keepdims=True)
    return (response / mean - 1.0) * 100.0
```

The dataset also contains the region of interest (ROI) of each voxel and the names of the ROIs.

```{code-cell} ipython3
roi_index = dataset.roi_index
roi_mapping = dataset.roi_mapping

roi_mapping
```

The ROIs are numerosity maps that were delineated on the cortical surface in a previous study. Each voxel has the ROI of the closest vertex of the surface. The names of the ROIs start with their hemisphere (`lh` for left and `rh` for right), followed by `Num` to indicate that they are numerosity maps. NumTO (temporal-occipital) and NumPO (parietal-occipital) lie in the occipital cortex, NumPC1, NumPC2, and NumPC3 in the parietal cortex, and NumF in the frontal cortex (see Hendrikx et al., 2024, for details). Voxels outside the maps are `unassigned`.

We only fit the valid voxels that lie inside a numerosity map. All analyses below interpret the results within the numerosity maps, so the voxels outside the maps are not needed. Moreover, fitting all valid voxels (about 390,000) takes a long time without a graphics processing unit (GPU), whereas the maps contain only about 26,500 of them. We store the indices of the selected voxels, so that we can project the results onto the cortical surface later.

```{code-cell} ipython3
roi_labels = [label for label in roi_mapping if label > 0]  # The numerosity maps (without -2 and -1)
roi_names = [roi_mapping[label] for label in roi_labels]

# Rows of the response that are valid and lie inside a numerosity map
fit_voxels = np.flatnonzero(is_valid & np.isin(roi_index, roi_labels))

pd.Series(roi_index[fit_voxels]).value_counts().sort_index().rename(index=roi_mapping)  # Voxels per map
```

### Projecting voxels onto the cortical surface

To show the results on the cortical surface, we need to know which voxels belong to each vertex of the surface `dataset.mesh`. The dataset provides this assignment in the file `dataset.files["surface_mapping_method2"]` in a compact form: `method2_vertex_voxel_rows` lists the voxels of all vertices one after another, and the voxels of vertex `v` are the entries from `offsets[v]` to `offsets[v + 1]` of this list, where `offsets` is `method2_vertex_voxel_offsets`. The voxels are given as rows of `dataset.response`, and the vertices are ordered with the left hemisphere first.

We write a function that projects one value per voxel onto the surface by averaging the values of the voxels of each vertex. Voxels without a value (NaN) are left out, and a vertex without any voxel with a value is NaN.

```{code-cell} ipython3
with np.load(dataset.files["surface_mapping_method2"]) as archive:
    offsets = archive["method2_vertex_voxel_offsets"]
    mapped_voxels = archive["method2_vertex_voxel_rows"]

num_vertices = dataset.mesh.n_vertices
mapped_vertices = np.repeat(np.arange(num_vertices), np.diff(offsets))  # The vertex of each entry


def project_to_surface(voxel_values: np.ndarray) -> np.ndarray:
    """Average one value per voxel (NaN for voxels without a value) over the voxels of each vertex."""
    values = voxel_values[mapped_voxels]
    has_value = ~np.isnan(values)
    total = np.bincount(mapped_vertices[has_value], weights=values[has_value], minlength=num_vertices)
    count = np.bincount(mapped_vertices[has_value], minlength=num_vertices)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def fill_voxels(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Put values for some rows of the response into an array with one value per voxel (NaN elsewhere)."""
    voxel_values = np.full(len(response), np.nan)
    voxel_values[rows] = values
    return voxel_values
```

To see where the maps lie, we show them on flattened surfaces (flatmaps) of both hemispheres, which show the entire cortex at once. A flat surface covers fewer vertices than the full cortical surface, because a part of it (e.g., the medial wall) was cut away to flatten it. The dataset therefore provides, for each hemisphere, the flat surface (e.g., `dataset.files["lh_flat"]`) and the index of each of its vertices on the full surface (e.g., `dataset.files["lh_vertex_indices"]`).

We write a function that builds a flat surface mesh with both hemispheres side by side and returns, for each vertex of the flat mesh, the index of the vertex on the full surface of both hemispheres (left first), which is the order of the values that `project_to_surface` returns. The orientation of a flat surface is arbitrary, so we rotate the hemispheres to show them upright.

```{code-cell} ipython3
import nibabel as nib
from nilearn.surface import InMemoryMesh
from nilearn.surface import PolyMesh


def rotate_xy(coords: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the x-y coordinates of a flat surface counter-clockwise by `angle` degrees around their center."""
    angle = np.deg2rad(angle)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = coords.copy()
    center = coords[:, :2].mean(axis=0)
    rotated[:, :2] = (coords[:, :2] - center) @ rotation.T
    return rotated


def make_flat_mesh(dataset, rotation: dict[str, float], gap: float = 20.0) -> tuple[PolyMesh, np.ndarray]:
    """Build a flat mesh of both hemispheres and the index of each of its vertices on the full surface."""
    parts, indices = {}, []
    row_offset = 0  # The vertices of the right hemisphere follow those of the left hemisphere

    for hemi, prefix in (("left", "lh"), ("right", "rh")):
        flat = nib.load(dataset.files[f"{prefix}_flat"])
        coords = rotate_xy(np.asarray(flat.darrays[0].data, dtype=np.float64), rotation[hemi])

        # Move the hemispheres apart along x, the left hemisphere to the left
        if hemi == "left":
            coords[:, 0] -= coords[:, 0].max() + gap / 2
        else:
            coords[:, 0] -= coords[:, 0].min() - gap / 2

        parts[hemi] = InMemoryMesh(coords, np.asarray(flat.darrays[1].data, dtype=np.int64))
        indices.append(np.load(dataset.files[f"{prefix}_vertex_indices"]) + row_offset)
        row_offset += dataset.mesh.parts[hemi].n_vertices

    return PolyMesh(**parts), np.concatenate(indices)


flat_mesh, flat_indices = make_flat_mesh(dataset, rotation={"left": -100, "right": -80})
```

The maps were delineated on the vertices of the surface, and the file `dataset.files["surface_metadata"]` contains the ROI of each vertex (`vertex_roi_index`), in the same order. We show the vertices that lie inside a map and receive at least one of the selected voxels with {py:func}`prfmodel.plotting.plot_surface_stat_map`, colored by their map. Each map exists in both hemispheres, so we give the same map the same color in both.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from prfmodel.plotting import plot_surface_stat_map

with np.load(dataset.files["surface_metadata"]) as archive:
    vertex_roi_index = archive["vertex_roi_index"]

# Vertices that receive at least one of the voxels that we fit
has_fit_voxel = ~np.isnan(project_to_surface(fill_voxels(1.0, fit_voxels)))

# Number of each map within its hemisphere (1 for NumTO to 6 for NumF); NaN for the other vertices
num_maps_per_hemisphere = len(roi_labels) // 2
is_map_vertex = has_fit_voxel & np.isin(vertex_roi_index, roi_labels)
map_surface = np.where(is_map_vertex, (vertex_roi_index - 1) % num_maps_per_hemisphere + 1, np.nan)

# One color per map: the first colors of "tab10", which do not include gray (the color of the other vertices)
map_cmap = ListedColormap(plt.get_cmap("tab10").colors[:num_maps_per_hemisphere])

fig, ax = plot_surface_stat_map(
    flat_mesh,
    map_surface[flat_indices],
    vmin=0.5,
    vmax=num_maps_per_hemisphere + 0.5,
    cmap=map_cmap,
    title="Numerosity maps",
    figsize=(12, 6),
)

# Label the colorbar with the map names (without the hemisphere prefix)
colorbar_ax = next(axis for axis in fig.axes if axis.name != "3d")
colorbar_ax.set_yticks(
    np.arange(1, num_maps_per_hemisphere + 1),
    labels=[name.split("_")[1] for name in roi_names[:num_maps_per_hemisphere]],
)
plt.show()
```


## Removing slow trends

BOLD responses often contain slow fluctuations that are not related to the stimulus. We remove these slow trends from each timecourse with a basis of cosine functions (i.e., a discrete cosine transform basis): We fit the cosine functions to each timecourse with least squares (using the pseudoinverse of the basis) and subtract the fitted trend. `N_DCT` sets the highest frequency of the cosine functions in cycles per timecourse. With `N_DCT = 0.5`, the basis contains a constant and a cosine that completes half a cycle over the 44 time frames. Later, we apply the same detrending to the model predictions, so that the observed and predicted responses are processed in the same way.

We keep the basis this small on purpose. The response to the stimulus itself changes slowly: A population that prefers a small numerosity responds at the beginning and the end of the cycle but not in the middle, which resembles a cosine that completes a full cycle over the 44 time frames. If the basis contained frequencies close to those of the response to the stimulus (e.g., with `N_DCT = 1.0`), detrending would remove part of the signal that we want to model.

```{code-cell} ipython3
N_DCT = 0.5

# Cosine basis with frequencies from 0 (a constant) to N_DCT in steps of half a cycle
time = np.linspace(0, 2 * np.pi, NUM_CYCLE_FRAMES)
frequencies = np.arange(0, N_DCT + 0.25, 0.5)
trends = np.cos(np.outer(time, frequencies))
pinv_trends = np.linalg.pinv(trends)


def detrend(response: np.ndarray) -> np.ndarray:
    """Fit the cosine basis to each timecourse and subtract the fitted trend."""
    return response - (response @ pinv_trends.T) @ trends.T


response_psc = percent_signal_change(response[fit_voxels])
response_fit = detrend(response_psc)

response_fit.shape
```

We compare the timecourse of a single voxel before and after detrending. The size of the slow trend differs between voxels, and some voxels have almost none. As an example, we pick a voxel whose trend is large relative to its signal (larger than in 90% of the voxels), so that the effect of detrending is clearly visible.

```{code-cell} ipython3
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8c8c8c"

# Size of the removed trend relative to the variation of the signal, for each voxel
fitted_trend = response_psc - response_fit
trend_size = np.abs(fitted_trend).max(axis=1) / response_psc.std(axis=1)
example = np.argsort(trend_size)[int(0.9 * len(trend_size))]

time_frames = np.arange(NUM_CYCLE_FRAMES)

fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
ax.axhline(0.0, color="#bdbdbd", linewidth=1)
ax.plot(time_frames, response_psc[example], color=BLUE, linewidth=2, label="Before detrending")
ax.plot(time_frames, fitted_trend[example], color=GRAY, linewidth=1.5, linestyle="--", label="Removed trend")
ax.plot(time_frames, response_fit[example], color=ORANGE, linewidth=2, label="After detrending")
ax.set(
    xlabel="Time frame (in TR)",
    ylabel="BOLD response (in PSC)",
    title=f"Voxel {fit_voxels[example]} ({roi_mapping[roi_index[fit_voxels[example]]]})",
)
ax.legend(frameon=False)
plt.show()
```

Detrending removes slow changes of the response over the cycle that are not related to the stimulus.

+++

## Inspecting the data

Before we model the BOLD response, we look at all timecourses at once in a heatmap with
{py:func}`prfmodel.plotting.plot_response_heatmap`. Note that the unit of the time frames is the repetition time (TR), which is 2.1 seconds for this dataset.

```{code-cell} ipython3
from prfmodel.plotting import plot_response_heatmap

# Sort the voxels by numerosity map (in the order of the mapping) so that each map forms one block of rows
by_map = np.argsort(roi_index[fit_voxels], kind="stable")

fig, ax = plot_response_heatmap(
    response_fit[by_map],
    vmin=-2,
    vmax=2,
    cmap="RdBu_r",
    xlabel="Time frame (in TR)",
    ylabel="Numerosity map",
    colorbar_label="BOLD response (in PSC)",
    figsize=(8, 7),
)

# Mark the borders between the maps and label each map in the middle of its rows
map_sizes = np.array([(roi_index[fit_voxels] == label).sum() for label in roi_labels])
map_ends = np.cumsum(map_sizes)
for end in map_ends[:-1]:
    ax.axhline(end - 0.5, color="black", linewidth=0.8)
ax.set_yticks(map_ends - map_sizes / 2, labels=roi_names)
plt.show()
```

A voxel that responds to numerosity should show two peaks per cycle, one during the ascending and one during the descending sequence, with a distance between the peaks that depends on its preferred numerosity (see above). Voxels whose responses do not follow the stimulus show noise without such peaks. The voxels are sorted by numerosity map, so the heatmap also shows whether the maps differ in how clearly their voxels follow the stimulus.

+++

## Defining the pRF model

Now that the stimulus and the BOLD response are in place, we can define a pRF model that *predicts* the response to the stimulus. We use the Gaussian 1D pRF model by Harvey et al. (2013), which predicts the response in three steps:

1. The pRF is a Gaussian in log-numerosity space with a center `mu` (the preferred _log numerosity_) and a size `sigma`. At each time frame, the model computes how strongly the shown numerosity drives the pRF.
2. This neural response is convolved with an impulse response that follows the shape of the hemodynamic response in the brain, which turns it into a predicted BOLD response.
3. A `baseline` and an `amplitude` parameter shift and scale the prediction.

We first look at the pRF itself. The figure shows two pRFs with different preferred numerosities together with the numerosities that the stimulus shows. Use the slider to change the pRF size `sigma` of both pRFs.

```{code-cell} ipython3
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook_connected"  # Requires internet connection to work
pio.templates.default = "simple_white"

numerosities = np.geomspace(0.5, 40.0, 300)
preferred_numerosities = [2.0, 6.0]
sigmas = np.round(np.arange(0.1, 1.51, 0.1), 1)
start = int(np.argmin(np.abs(sigmas - 0.3)))  # Slider position when the figure is first shown


def tuning_curve(preferred: float, sigma: float) -> np.ndarray:
    """Gaussian pRF in log-numerosity space, relative to its peak."""
    return np.exp(-((np.log(numerosities) - np.log(preferred)) ** 2) / (2 * sigma**2))


# Vertical lines at the numerosities of the stimulus, drawn as one trace (separated by None) that stays fixed
stimulus_lines = go.Scatter(
    x=np.repeat(unique_numerosities, 3),
    y=np.tile([0.0, 1.05, None], len(unique_numerosities)),
    mode="lines",
    line={"color": "#bdbdbd", "width": 1},
    name="Numerosities of the stimulus",
    hoverinfo="skip",
)
prf_curves = [
    go.Scatter(
        x=numerosities,
        y=tuning_curve(preferred, sigmas[start]),
        mode="lines",
        line={"color": color, "width": 2},
        name=f"Preferred numerosity {preferred:.0f}",
        hovertemplate="Numerosity %{x:.1f}<br>Tuning %{y:.2f}<extra></extra>",
    )
    for preferred, color in zip(preferred_numerosities, [BLUE, ORANGE], strict=True)
]

fig = go.Figure(
    data=[stimulus_lines, *prf_curves],
    # Each frame only updates the pRF curves (traces 1 and 2)
    frames=[
        go.Frame(
            name=f"{sigma:.1f}",
            traces=[1, 2],
            data=[go.Scatter(y=tuning_curve(preferred, sigma)) for preferred in preferred_numerosities],
        )
        for sigma in sigmas
    ],
)
fig.update_xaxes(
    type="log",
    range=np.log10([numerosities[0], numerosities[-1]]),  # Ranges of log axes are given in log10 units
    tickvals=unique_numerosities,
    ticktext=unique_numerosities.astype(int),
    title="Numerosity (log scale)",
)
fig.update_yaxes(range=[0.0, 1.05], title="Tuning (relative to peak)")
fig.update_layout(
    height=450,
    legend={"orientation": "h", "y": 1.1},
    sliders=[
        {
            "active": start,
            "currentvalue": {"prefix": "pRF size (sigma): "},
            "pad": {"t": 50},
            "steps": [
                {
                    "method": "animate",
                    "label": f"{sigma:.1f}",
                    "args": [
                        [f"{sigma:.1f}"],
                        {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}},
                    ],
                }
                for sigma in sigmas
            ],
        },
    ],
)
fig.show()
```

On the log scale, each pRF is symmetric around its preferred numerosity. On the natural scale, it extends further towards larger numerosities, so a pRF distinguishes small numerosities (e.g., 2 and 3) better than large ones (e.g., 6 and 7). A narrow pRF (small `sigma`) responds to only a few of the numerosities in the stimulus, a wide pRF to many.

The impulse response describes how the BOLD response follows a short burst of neural activity. Its parameters default to the Glover hemodynamic response function (see
{py:func}`~prfmodel.impulse.defaults.default_two_gamma_impulse_glover_hrf`), so we only need to set `weight_deriv`. This parameter weights the temporal derivative of the impulse response and absorbs small differences in the timing of the BOLD response between voxels. Each time frame is one TR of 2.1 seconds, which the stimulus stores as its `frame_duration`. The impulse response is sampled at this frame duration, so the prediction has the same sampling rate as the observed responses (see also the section [](../../important_details.md)). We plot the impulse response for three values of `weight_deriv`. On its own, the impulse model does not know the stimulus, so we sample it at the frame duration of the stimulus with {py:meth}`~prfmodel.impulse.base.BaseImpulse.at_resolution`.

```{code-cell} ipython3
from prfmodel.impulse import DerivativeTwoGammaImpulse

# Without a resolution, the impulse model is sampled at the frame duration of the stimulus
impulse_model = DerivativeTwoGammaImpulse()

weight_derivs = [-0.5, 0.0, 0.5]
impulse_response = np.asarray(
    impulse_model.at_resolution(stimulus.frame_duration)(pd.DataFrame({"weight_deriv": weight_derivs})),
)
impulse_time = np.arange(impulse_response.shape[1]) * stimulus.frame_duration

fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
ax.axhline(0.0, color="#bdbdbd", linewidth=1)
for weight_deriv, response_curve, color in zip(weight_derivs, impulse_response, [BLUE, GRAY, ORANGE], strict=True):
    ax.plot(impulse_time, response_curve, color=color, linewidth=2, label=f"weight_deriv = {weight_deriv}")
ax.set(xlabel="Time (in s)", ylabel="Impulse response", title="Impulse response")
ax.legend(frameon=False)
plt.show()
```

Negative values of `weight_deriv` shift the peak of the impulse response earlier in time and positive values later. We start with `weight_deriv = -0.5` and estimate it for each voxel during fitting.

We insert the impulse model into the {py:class}`prfmodel.models.prf.Gaussian1DPRFModel`, which combines all three steps, and predict the response of a pRF that prefers numerosity 3 to the full stimulus.

```{code-cell} ipython3
from prfmodel.models.prf import Gaussian1DPRFModel

prf_model = Gaussian1DPRFModel(impulse_model=impulse_model)

example_params = pd.DataFrame(
    {"mu": [np.log(3.0)], "sigma": [0.5], "weight_deriv": [-0.5], "baseline": [0.0], "amplitude": [1.0]},
)

# Neural response: the pRF tuning (relative to its peak) at the numerosity that is shown at each time frame
tuning_at_numerosities = np.exp(-((unique_log_numerosities - np.log(3.0)) ** 2) / (2 * 0.5**2))
neural_response = stimulus.design @ tuning_at_numerosities

# Predicted BOLD response: the output of the model
bold_prediction = np.asarray(prf_model(stimulus, example_params))[0]

# Time frames at which numerosity 3 is shown (the design column of numerosity 3 is one)
column_of_3 = np.flatnonzero(unique_numerosities == 3)[0]
frames_with_3 = np.flatnonzero(stimulus.design[:, column_of_3] == 1)
all_frames = np.arange(stimulus.design.shape[0])

fig, (ax_neural, ax_bold) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, layout="constrained")

for ax in (ax_neural, ax_bold):
    ax.axvspan(-0.5, NUM_PRESCAN_FRAMES - 0.5, color="#bdbdbd", alpha=0.5, label="Pre-scan interval")
    for i, frame in enumerate(frames_with_3):
        ax.axvspan(frame - 0.5, frame + 0.5, color=ORANGE, alpha=0.3, label="Numerosity 3 shown" if i == 0 else None)
    for cycle in range(1, NUM_CYCLES):
        ax.axvline(NUM_PRESCAN_FRAMES + cycle * NUM_CYCLE_FRAMES - 0.5, color=GRAY, linestyle="--", linewidth=1)

ax_neural.step(all_frames, neural_response, where="mid", color=BLUE, linewidth=2)
ax_neural.set(ylabel="Neural response\n(relative to peak)", title="Predictions for a pRF that prefers numerosity 3")
ax_neural.legend(frameon=False, loc="upper right")

ax_bold.plot(all_frames, bold_prediction, color=BLUE, linewidth=2)
ax_bold.set(xlabel="Time frame (in TR; dashed lines: start of a cycle)", ylabel="Predicted BOLD\nresponse")

# Show the pre-scan interval and the first cycle; the other cycles repeat the same pattern
ax_bold.set_xlim(-0.5, NUM_PRESCAN_FRAMES + NUM_CYCLE_FRAMES - 0.5)

plt.show()
```

The figure shows the pre-scan interval and the first cycle; the other three cycles repeat the same pattern. The top
panel shows the neural response. The model describes the stimulus with one numerosity per time frame, so the neural
response is constant within a time frame: It is the pRF evaluated at the shown numerosity. The model assumes that
neurons respond without delay, so the neural response is highest exactly when numerosity 3 is shown (orange). The pRF
also responds to the neighboring numerosities, so the response builds up before and fades out after numerosity 3.

The bottom panel shows the prediction of the model, the BOLD response. It is the neural response convolved with the
impulse response, which smooths it and delays it by about two time frames (roughly 4 seconds), the peak time of the
impulse response above. This delay is the only delay in the model.

### Matching the predictions to the preprocessed responses

The pRF model predicts the response to the full stimulus with 182 time frames, but the observed responses only contain a single averaged cycle of 44 time frames. To compare them, we process the predictions in the same way as the observed responses:

1. We remove the six time frames of the pre-scan interval. We do not remove them from the stimulus itself: The subject already saw the stimulus during the pre-scan interval, and because the BOLD response is slow, this affects the response at the beginning of the first cycle.
2. We average the four cycles in the remaining 176 time frames.
3. We detrend the averaged predictions with the same cosine basis as the observed responses.

We implement these steps in a function that wraps the prediction function of the model (i.e., its `call` method) and replace the original prediction function with the wrapped one. This way, the fitters below automatically compare the processed predictions against the observed responses.

```{code-cell} ipython3
from collections.abc import Callable

from keras import ops
from prfmodel.stimuli import PRFStimulus
from prfmodel.typing import Tensor


def average_cycles_and_detrend(fn: Callable) -> Callable:
    """Wrap a prediction function to remove the pre-scan frames, average the cycles, and detrend the predictions."""

    def wrapper(stimulus: PRFStimulus, parameters: pd.DataFrame, *args, **kwargs) -> Tensor:
        predictions = fn(stimulus, parameters, *args, **kwargs)

        # 1. Remove the pre-scan frames
        predictions = predictions[:, NUM_PRESCAN_FRAMES:]

        # 2. Average the repeated stimulus cycles
        predictions = ops.mean(ops.reshape(predictions, (-1, NUM_CYCLES, NUM_CYCLE_FRAMES)), axis=1)

        # 3. Fit the trends to each prediction and subtract them (same basis as for the observed responses)
        beta = ops.matmul(predictions, ops.transpose(ops.cast(pinv_trends, predictions.dtype)))
        fitted_trend = ops.matmul(beta, ops.transpose(ops.cast(trends, predictions.dtype)))

        return predictions - fitted_trend

    return wrapper


prf_model.call = average_cycles_and_detrend(prf_model.call)
```

To see what the processed predictions look like, we predict the response of pRFs that prefer different numerosities.

```{code-cell} ipython3
num_unique = len(unique_log_numerosities)

start_params = pd.DataFrame(
    {
        "mu": unique_log_numerosities,
        "sigma": [0.5] * num_unique,
        "weight_deriv": [-0.5] * num_unique,
        "baseline": [0.0] * num_unique,
        "amplitude": [1.0] * num_unique,
    },
)

simulated_response = np.asarray(prf_model(stimulus, start_params))

fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
for numerosity, prediction in zip(unique_numerosities.astype(int), simulated_response, strict=True):
    ax.plot(time_frames, prediction, linewidth=1.5, label=str(numerosity))
ax.set(xlabel="Time frame (in TR)", ylabel="Predicted response", title="Predicted response for one stimulus cycle")
ax.legend(title="Preferred\nnumerosity", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.show()
```

The preferred numerosity `mu` sets the timing of the peaks: pRFs that prefer small numerosities respond at the beginning and the end of the cycle, when small numerosities are shown, whereas pRFs that prefer larger numerosities respond closer to the middle of the cycle, around the baseline.

The pRF size `sigma` sets how sharply a pRF is tuned. We compare a narrow and a wide pRF that both prefer numerosity 3.

```{code-cell} ipython3
sigmas = [0.2, 2.0]

sigma_params = pd.DataFrame(
    {
        "mu": [np.log(3.0)] * len(sigmas),
        "sigma": sigmas,
        "weight_deriv": [-0.5] * len(sigmas),
        "baseline": [0.0] * len(sigmas),
        "amplitude": [1.0] * len(sigmas),
    },
)

sigma_response = np.asarray(prf_model(stimulus, sigma_params))

fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
for sigma, prediction, color in zip(sigmas, sigma_response, [BLUE, ORANGE], strict=True):
    ax.plot(time_frames, prediction, color=color, linewidth=2, label=f"sigma = {sigma}")
ax.set(
    xlabel="Time frame (in TR)",
    ylabel="Predicted response",
    title="Predicted response for a preferred numerosity of 3",
)
ax.legend(frameon=False)
plt.show()
```

A narrow pRF only responds to numerosities close to 3, which gives two distinct peaks per cycle. A wide pRF responds to many numerosities, so its response changes less over the cycle and its peaks merge. Because a wide pRF predicts an almost flat response, its preferred numerosity is hard to estimate from the data.

+++

## Fitting the pRF model

We fit the pRF model to the detrended responses in three stages. We begin with a grid search to find good values for the parameters of interest (`mu` and `sigma`). Then, we use least squares to estimate the `baseline` and `amplitude`.Finally, we use stochastic gradient descent (SGD) to finetune all parameters at once.

### Grid search

We define ranges of `mu` and `sigma` to construct a grid of parameter values from. We create them from the stimulus with {py:func}`prfmodel.fitters.grid_values_1d_prf`. For `weight_deriv`, `baseline`, and `amplitude`, we only provide a single value so that they stay constant across the entire grid.

```{code-cell} ipython3
from prfmodel.fitters import grid_values_1d_prf

param_ranges = grid_values_1d_prf(stimulus) | {
    # delay, dispersion, undershoot, u_dispersion, and ratio use the default Glover HRF parameters
    "weight_deriv": [-0.5],
    "baseline": [0.0],
    "amplitude": [1.0],
}

{name: len(values) for name, values in param_ranges.items()}  # Number of values for each parameter
```

By default, the function creates 49 values for `mu` that span twice the range of the stimulus around its midpoint and 50 log-spaced values for `sigma` that range from the smallest spacing between the stimulus coordinates to their full range, which gives $49 \times 50 = 2450$ parameter combinations. This is still a relatively small grid and we recommend specifying finer grids in practice.

Two properties of the stimulus bound what these ranges can achieve. First, the range of `mu` extends beyond the displayed numerosities, from about 0.2 to 90 on the natural scale. Estimates outside the numerosities 1 to 7 (numerosity 20 is only shown as a baseline) are only weakly constrained by the data, and we will exclude them when we analyze the pRF parameters. Extending the grid beyond the stimulus prevents such voxels from piling up at the edge of the grid.
Second, the stimulus samples log-numerosity space at only eight points, the closest of which are $\log(7) - \log(6) \approx 0.15$ apart. This spacing is the smallest value of `sigma` in the grid. A pRF much narrower than the spacing responds to a single numerosity no matter how small `sigma` becomes, so estimates near the smallest value should be read as "no wider than one stimulus level".

We construct the {py:class}`prfmodel.fitters.GridFitter` and perform the grid search. We set `batch_size=20` to evaluate 20 parameter combinations at the same time, which saves memory. By default, the `loss` is the negative correlation, which ignores differences in baseline and amplitude between predictions and observed data, so the grid search cannot estimate `baseline` and `amplitude`.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter

grid_fitter = GridFitter(
    model=prf_model,
    stimulus=stimulus,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

grid_history, grid_params = grid_fitter.fit(
    data=response_fit,
    parameter_values=param_ranges,
    batch_size=20,
)

grid_params
```

### Least squares

We estimate `baseline` and `amplitude` with least squares by regressing the observed responses on the predicted responses of each voxel. Because the observed and the predicted responses are both detrended, which also removes their mean, the `baseline` estimates are close to zero.

```{code-cell} ipython3
from prfmodel.fitters import LeastSquaresFitter

ls_fitter = LeastSquaresFitter(
    model=prf_model,
    stimulus=stimulus,
)

ls_history, ls_params = ls_fitter.fit(
    data=response_fit,
    parameters=grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)

ls_params
```

### Stochastic gradient descent

To finetune all parameters, we use the {py:class}`~prfmodel.fitters.SGDFitter` with the least-squares parameters as starting values. Because `sigma` is strictly positive, we include an adapter that lets the fitter optimize `sigma` on an unconstrained log scale and transforms it back before making predictions.

By default, {py:class}`~prfmodel.fitters.SGDFitter` optimizes every parameter it is given, so `weight_deriv` is estimated per voxel as well. Pass `fixed_parameters=["weight_deriv"]` to `fit` to keep the impulse response identical across voxels instead.

```{code-cell} ipython3
from prfmodel.fitters import SGDFitter
from prfmodel.fitters.adapter import Adapter, ParameterTransform

adapter = Adapter([ParameterTransform(["sigma"], transform_fun=ops.log, inverse_fun=ops.exp)])

sgd_fitter = SGDFitter(
    model=prf_model,
    stimulus=stimulus,
    adapter=adapter,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
)

sgd_history, sgd_params = sgd_fitter.fit(
    data=response_fit,
    init_parameters=ls_params,
)

sgd_params
```

## Evaluating the model fit

We compare the model predictions against the observed responses. We wrap the model in {py:func}`prfmodel.utils.batched` to make predictions for batches of voxels sequentially, which saves memory. We quantify the fit with the R-squared, computed with {py:func}`prfmodel.utils.calculate_r_squared`, which is the proportion of variance in the observed responses explained by the predictions. Because we compare the predictions against the responses that we fitted the model to, the scores describe the in-sample fit.

```{code-cell} ipython3
from prfmodel.utils import batched, calculate_r_squared

prf_model_batched = batched(prf_model)

pred_response = np.asarray(prf_model_batched(stimulus, sgd_params, batch_size=100))
r_squared = calculate_r_squared(response_fit, pred_response)  # One score per voxel

r_squared.shape
```

We look at the distribution of the scores with {py:func}`prfmodel.plotting.plot_r_squared_hist`. We clip the scores to $[0, 1]$: R-squared is negative whenever a prediction fits worse than the mean of the data, and those voxels end up in the leftmost bin.

```{code-cell} ipython3
from prfmodel.plotting import plot_r_squared_hist

plot_r_squared_hist(r_squared, bins=20, value_range=(0.0, 1.0), clip=True);
```

We also compare the observed and predicted timecourses of the best-fitting voxel in each numerosity map. Use the slider to switch between the maps. We only consider voxels whose preferred numerosity lies between 1 and 7, the numerosities that the stimulus shows (we explain this choice below).

```{code-cell} ipython3
numerosity_estimates = np.exp(sgd_params["mu"].to_numpy())
is_in_stimulus_range = (numerosity_estimates >= 1) & (numerosity_estimates <= 7)

# Best-fitting voxel in each map (maps without a voxel in the stimulus range are left out)
best_voxels = {}
for label in roi_labels:
    in_roi = np.flatnonzero((roi_index[fit_voxels] == label) & is_in_stimulus_range)
    if in_roi.size > 0:
        best_voxels[label] = in_roi[np.argmax(r_squared[in_roi])]


def fit_title(label: int) -> str:
    """Describe the best-fitting voxel of a map."""
    best = best_voxels[label]
    return (
        f"{roi_mapping[label]}: R-squared = {r_squared[best]:.2f}, "
        f"preferred numerosity = {numerosity_estimates[best]:.1f}"
    )


# Use the same y-axis range for all maps, so that their responses can be compared
selected = list(best_voxels.values())
y_limit = 1.05 * max(np.abs(response_fit[selected]).max(), np.abs(pred_response[selected]).max())

first_label, first = next(iter(best_voxels.items()))
fig = go.Figure(
    data=[
        go.Scatter(
            x=time_frames,
            y=response_fit[first],
            mode="lines",
            line={"color": GRAY, "width": 1.5},
            name="Observed",
        ),
        go.Scatter(
            x=time_frames,
            y=pred_response[first],
            mode="lines",
            line={"color": BLUE, "width": 2},
            name="Predicted",
        ),
    ],
    frames=[
        go.Frame(
            name=roi_mapping[label],
            data=[go.Scatter(y=response_fit[best]), go.Scatter(y=pred_response[best])],
            layout={"title": {"text": fit_title(label)}},
        )
        for label, best in best_voxels.items()
    ],
)
fig.add_hline(y=0.0, line={"color": "#bdbdbd", "width": 1}, layer="below")
fig.update_xaxes(range=[time_frames[0], time_frames[-1]], title="Time frame (in TR)")
fig.update_yaxes(range=[-y_limit, y_limit], title="BOLD response (in PSC)")
fig.update_layout(
    title={"text": fit_title(first_label)},
    height=450,
    legend={"orientation": "h", "y": 1.02, "x": 1.0, "xanchor": "right"},
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "Map: "},
            "pad": {"t": 50},
            "steps": [
                {
                    "method": "animate",
                    "label": roi_mapping[label],
                    "args": [
                        [roi_mapping[label]],
                        {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}},
                    ],
                }
                for label in best_voxels
            ],
        },
    ],
)
fig.show()
```

+++

## Interpreting the model parameters

To interpret the pRF parameters, we zoom in on voxels for which we can trust the estimates. We select voxels that fulfill three criteria:

- The R-squared is above 0.2 (note that this threshold is somewhat arbitrary).
- The preferred numerosity lies between 1 and 7, the numerosities that the stimulus shows (excluding the baseline 20).
- The pRF size (`sigma`) is below 2.8, which is close to the range of the log numerosities in the stimulus
  ($\log(20) \approx 3.0$). Such a wide Gaussian predicts an almost flat response, so its center `mu` cannot be
  identified. Note that SGD optimizes `sigma` on an unconstrained log scale, so the estimates are no longer limited by
  the grid.

We define the selection as a function because we apply it again with a stricter R-squared threshold when we look at a single numerosity map below.

```{code-cell} ipython3
def select_voxels(
    params: pd.DataFrame,
    r_squared: np.ndarray,
    r_squared_threshold: float = 0.2,
) -> np.ndarray:
    """Select voxels with a good fit, a preferred numerosity between 1 and 7, and an identifiable pRF center."""
    numerosity = np.exp(params["mu"].to_numpy())
    return (
        (r_squared > r_squared_threshold)
        & (numerosity >= 1)
        & (numerosity <= 7)
        & (params["sigma"].to_numpy() < 2.8)
    )


is_selected = select_voxels(sgd_params, r_squared)

is_selected.mean()  # Fraction of fitted voxels that are selected
```

We add the name of the numerosity map to the parameters of each voxel and convert `mu` into the preferred numerosity on the natural scale.

```{code-cell} ipython3
final_params = sgd_params.copy()
final_params["roi"] = pd.Categorical(
    [roi_mapping[label] for label in roi_index[fit_voxels]],
    categories=roi_names,  # Keep the order of the maps from the mapping
    ordered=True,
)
final_params["numerosity"] = np.exp(final_params["mu"])

params_selected = final_params.loc[is_selected]

params_selected.groupby("roi", observed=False)[["numerosity", "sigma"]].agg(["mean", "std", "count"]).round(2)
```

We plot the average preferred numerosity and pRF size of each map with {py:func}`prfmodel.plotting.plot_parameter_by_roi`. The error bars show the standard deviation across voxels (not the standard error of the mean).

```{code-cell} ipython3
from prfmodel.plotting import plot_parameter_by_roi

fig, ax = plot_parameter_by_roi(
    params_selected,
    "numerosity",
    params_selected["roi"],
    order=roi_names,
    xlabel="Numerosity map",
    ylabel="Preferred numerosity",
    figsize=(12, 4.5),
)
ax.tick_params(axis="x", labelrotation=45)
plt.show()
```

```{code-cell} ipython3
fig, ax = plot_parameter_by_roi(
    params_selected,
    "sigma",
    params_selected["roi"],
    order=roi_names,
    xlabel="Numerosity map",
    ylabel="pRF size (sigma, in log numerosity)",
    figsize=(12, 4.5),
)
ax.tick_params(axis="x", labelrotation=45)
plt.show()
```

When comparing the maps, keep in mind the variation within each map (the error bars) and the number of selected voxels in each map (see the table above): A difference between two maps is only meaningful if it is large relative to the variation within the maps, and an average over few voxels is less reliable than an average over many.

+++

## Preferred numerosity on the cortical surface

We can also look at the preferred numerosity on the cortical surface, using the flat mesh (`flat_mesh` and `flat_indices`) that we built above to show the numerosity maps. We project the preferred numerosity of the selected voxels onto the surface with `project_to_surface`, so each vertex shows the average over its selected voxels (NaN for vertices without any), and plot it on the flat mesh with {py:func}`prfmodel.plotting.plot_surface_stat_map`.

```{code-cell} ipython3
from prfmodel.plotting import plot_surface_stat_map

numerosity_surface = project_to_surface(fill_voxels(params_selected["numerosity"], fit_voxels[is_selected]))

plot_surface_stat_map(
    flat_mesh,
    numerosity_surface[flat_indices],
    vmin=1.0,
    vmax=7.0,
    cmap="viridis",
    title="Preferred numerosity",
    figsize=(12, 6),
);
```

Each colored patch is a numerosity map. Numerosity maps are topographic: Neighboring vertices prefer similar numerosities, so the preferred numerosity should change gradually from one end of a map to the other rather than randomly between vertices. We quantify this change along a single map in the next section.

+++

## Preferred numerosity along a numerosity map

To measure positions within a map, we use the surface again. The dataset contains the geodesic distance of each vertex within its map in mm, which is measured along the cortical surface rather than in a straight line through the brain. The distance is 0 mm at the end of the map with low preferred numerosities and increases towards the end with high preferred numerosities. The file `dataset.files["roi_data"]` stores it for each map as an array with one value per vertex (`roi_<label>_distances_mm`, NaN outside the map). We look at the right NumTO.

```{code-cell} ipython3
roi_label = 7  # rh_NumTO

with np.load(dataset.files["roi_data"]) as archive:
    roi_distance = archive[f"roi_{roi_label}_distances_mm"]

roi_mapping[roi_label]
```

We select the fitted voxels of the map with the same criteria as above, but with a stricter R-squared threshold of 0.3, and project their preferred numerosity onto the surface. We keep the vertices of the map that receive at least one selected voxel, together with their geodesic distance.

```{code-cell} ipython3
in_roi = roi_index[fit_voxels] == roi_label
keep = in_roi & select_voxels(sgd_params, r_squared, r_squared_threshold=0.3)

numerosity_vertices = project_to_surface(fill_voxels(np.exp(sgd_params["mu"].to_numpy()[keep]), fit_voxels[keep]))
has_distance = ~np.isnan(roi_distance) & ~np.isnan(numerosity_vertices)

distance_data = pd.DataFrame(
    {
        "distance": roi_distance[has_distance],
        "numerosity": numerosity_vertices[has_distance],
    },
)

len(distance_data)  # Number of vertices with a distance and a preferred numerosity
```

We group the vertices into distance bins of 2 mm and compute the mean preferred numerosity of each bin.

```{code-cell} ipython3
BIN_WIDTH = 2.0  # in mm

bin_edges = np.arange(0.0, distance_data["distance"].max() + BIN_WIDTH, BIN_WIDTH)


def bin_numerosity(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean preferred numerosity and its standard error in each distance bin."""
    bins = pd.cut(data["distance"], bins=bin_edges, include_lowest=True)
    binned = data.groupby(bins, observed=True)["numerosity"].agg(["mean", "sem", "count"]).reset_index()
    binned["distance"] = [interval.mid for interval in binned["distance"]]
    return binned


binned_data = bin_numerosity(distance_data)
```

We fit a logarithmic function to the mean preferred numerosity of the bins. To estimate the uncertainty of this fit, we use bootstrap resampling: We sample vertices with replacement, recompute the bins, and fit the logarithmic function again. We repeat this 1,000 times. The 2.5th and 97.5th percentiles of the resulting curves give a 95% confidence interval.

```{code-cell} ipython3
from scipy.optimize import curve_fit


def log_function(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Logarithmic function of the distance."""
    return a * np.log(x + c) + b


def fit_log_function(binned: pd.DataFrame, p0: tuple[float, ...] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Fit the logarithmic function to the mean preferred numerosity of the distance bins."""
    params, _ = curve_fit(
        log_function,
        binned["distance"],
        binned["mean"],
        p0=p0,
        bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 100.0]),
        maxfev=10000,
    )
    return params


fit_params = fit_log_function(binned_data)

rng = np.random.default_rng(22)
x_fit = np.linspace(distance_data["distance"].min(), distance_data["distance"].max(), 200)
bootstrap_curves = []

for _ in range(1000):
    resampled = distance_data.sample(len(distance_data), replace=True, random_state=rng)
    resampled_binned = bin_numerosity(resampled)
    if len(resampled_binned) < 3:  # Too few bins for three parameters
        continue
    try:
        bootstrap_curves.append(log_function(x_fit, *fit_log_function(resampled_binned, p0=fit_params)))
    except RuntimeError:  # The fit did not converge
        continue

lower, upper = np.percentile(bootstrap_curves, [2.5, 97.5], axis=0)

len(bootstrap_curves)  # Number of successful bootstrap fits
```

Finally, we plot the mean preferred numerosity of the bins. The error bars show the standard error of the mean within each bin. The solid line shows the logarithmic fit and the dashed lines its 95% bootstrap confidence interval.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

ax.errorbar(
    binned_data["distance"],
    binned_data["mean"],
    yerr=binned_data["sem"],
    fmt="o",
    color="black",
    capsize=3,
    label="Mean of distance bin",
    zorder=5,
)
ax.plot(x_fit, log_function(x_fit, *fit_params), color="black", linewidth=2.5, label="Logarithmic fit", zorder=4)
ax.plot(x_fit, lower, color="black", linestyle="--", linewidth=1)
ax.plot(x_fit, upper, color="black", linestyle="--", linewidth=1)

ax.set(
    xlabel="Geodesic distance (in mm)",
    ylabel="Preferred numerosity",
    ylim=(1, 7),
    title=f"Preferred numerosity along {roi_mapping[roi_label]}",
)
ax.legend(frameon=False)
plt.show()
```

In a topographic numerosity map, the preferred numerosity increases with the distance from the low end of the map. The logarithmic function describes an increase that is steep near the low end and flattens towards the high end, so that small numerosities take up more of the map than large ones.

### Controlling for other visual properties of the stimulus

Changing the number of dots also changes other visual properties of the stimulus, for example, the total area covered by the dots. Consequently, a neuron population whose response appears to be tuned to numerosity might instead be tuned to one of these properties. To control for this, the dataset contains responses to four stimulus conditions, each of which keeps a different visual property constant across numerosities:

- **Area (constant area):** The total area of all dots is the same for all numerosities.
- **Size (constant dot size):** The size of each dot is the same for all numerosities.
- **Dens (high density):** The same dots as in the constant area condition are shown in a smaller region of the screen,
  which increases their density.
- **Circ (constant circumference):** The total circumference of all dots is the same for all numerosities.

The `average` condition that we fitted above is the average of these four conditions. The conditions make the properties testable. Consider a population that responds to the total area of the dots rather than to their number: In the constant area and high density conditions, its response would not change with numerosity, because the total area stays the same. In the constant dot size condition, the total area grows with numerosity, so the population would appear to be tuned to numerosity. A population that is tuned to numerosity, in contrast, responds in the same way in all conditions. Thus, if a map represents numerosity, the preferred numerosity should change in the same way along the map in all four conditions.

The responses of each condition are loaded with the `task` option of {py:func}`~prfmodel.examples.load_dataset`. All conditions share the same voxels in the same order, so the ROI labels, the projection onto the surface, the geodesic distances, and the model from above apply to every condition. Only the valid voxels differ, because a voxel can fall outside the scanned volume in some runs of one condition but not of another. For example, the following code fits the constant area condition in the same map; repeating the steps of this section for its estimates gives the preferred numerosity along the map in this condition. We do not run it here because fitting all four conditions takes a while.


```python
area = load_dataset("numerosity-four-conditions", task="area")

# Voxels of the map that are valid in this condition
area_voxels = np.flatnonzero((roi_index == roi_label) & area.units["valid"].to_numpy())
area_response_fit = detrend(percent_signal_change(area.response[area_voxels]))

_, area_grid_params = grid_fitter.fit(data=area_response_fit, parameter_values=param_ranges, batch_size=20)
_, area_ls_params = ls_fitter.fit(
    data=area_response_fit,
    parameters=area_grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=200,
)
_, area_params = sgd_fitter.fit(data=area_response_fit, init_parameters=area_ls_params)

area_r_squared = calculate_r_squared(
    area_response_fit, np.asarray(prf_model_batched(stimulus, area_params, batch_size=100))
)
```

+++

## Conclusion

This example showed how to fit a one-dimensional Gaussian pRF model to fMRI data from a numerosity experiment. We loaded the responses of the voxels in a brain mask, converted them to percent signal change, removed slow trends, and adapted the model predictions to this preprocessing by wrapping the prediction function of the model.

We optimized the model parameters of the voxels in the numerosity maps with a grid search, least squares, and stochastic gradient descent, evaluated the model fit, and compared the estimated parameters between numerosity maps.

Finally, we projected the preferred numerosity from the voxels onto the cortical surface and showed it on flattened surfaces, investigated how it changes along a numerosity map, and showed how the four stimulus conditions of the dataset control for other visual properties of the dots.

+++

## References

Harvey, B. M., Klein, B. P., Petridou, N., & Dumoulin, S. O. (2013). Topographic representation of numerosity in the
human parietal cortex. *Science*, *341*(6150), 1123–1126. https://doi.org/10.1126/science.1239052

Hendrikx, E., Paul, J. M., van Ackooij, M., van der Stoep, N., & Harvey, B. M. (2024). Cortical quantity
representations of visual numerosity and timing overlap increasingly into superior cortices but remain distinct.
*NeuroImage*, *286*, 120515. https://doi.org/10.1016/j.neuroimage.2024.120515

Mauro, K. (2026). Four condition numerosity dataset. *figshare*. https://doi.org/10.6084/m9.figshare.34898940
