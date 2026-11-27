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

# Selecting the hemodynamic response function for each vertex in fMRI data from a visual experiment

+++

**Author**: Blazej M. Baczkowski (b.baczkowski@esciencecenter.nl)

**Difficulty**: Intermediate

+++

A population receptive field (pRF) model is usually fitted with the same hemodynamic response function (HRF) for every vertex, for example the Glover HRF. However, the shape of the hemodynamic response differs between brain areas. In this example, we fit a 2D Gaussian pRF model to fMRI data from a visual experiment and select the best fitting HRF for each vertex from a library of 20 canonical HRFs from GLMsingle (Prince et al., 2022). We then compare the selected HRFs between visual areas and compare the fit with that of a model with the Glover HRF.

See the tutorial {doc}`How the hemodynamic response function shapes the predicted response </tutorials/tutorials/hrf_selection>` for an introduction to the HRF library and the {py:class}`~prfmodel.impulse.HRFLibraryImpulse` model.

+++

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
    msg = 'Could not find the tensorflow package. Please install it with: pip install "prfmodel[tensorflow]"'
    raise ImportError(msg)
```

## Loading the dataset

We use the `7t-ncsf-visual` dataset (Roelofzen et al., 2025), which contains the response of a single subject measured in a 7T scanner with a repetition time (TR) of 1.5 seconds. We load the `prf` task, in which bars filled with natural images sweep across the screen in eight directions. The response has been detrended, converted to percent signal change (PSC), averaged over runs, and projected onto the HCP fs_LR 59k surface.

```{code-cell} ipython3
from prfmodel.examples import load_dataset

# Downloads on first use and caches in a user data directory (see prfmodel.examples.get_data_dir)
dataset = load_dataset("7t-ncsf-visual", task="prf", hemisphere="both")

response = dataset.response
stimulus = dataset.stimulus

print(response.shape)  # (num_vertices, num_frames)
print(stimulus)
```

The eight bar directions matter for HRF selection: a response that peaks later than the model predicts can also be explained by a pRF that is shifted along the direction of the bar. Because the bars sweep in opposite directions, such a shift cannot explain the response to all sweeps, which helps to separate the HRF from the pRF center.

For plotting, we load the flat surface of the `hcp-999999-surface` dataset, which has the same vertices as the response.

```{code-cell} ipython3
surface = load_dataset("hcp-999999-surface", surface_type="flat")
mesh = surface.mesh
```

## Selecting the vertices in visual areas

Each vertex carries the index of a visual area from the Benson et al. (2014) template atlas of the subject. The mapping from indices to area names comes with the dataset.

```{code-cell} ipython3
roi_mapping = dataset.roi_mapping
roi_mapping
```

We only fit the vertices that lie in one of the template areas and have a valid response, that is, a finite response that is not constant.

```{code-cell} ipython3
import numpy as np

area_order = [name for index, name in sorted(roi_mapping.items()) if index > 0]

is_in_area = dataset.roi_index > 0
is_valid = np.all(np.isfinite(response), axis=1) & (response.std(axis=1) > 0.0)
vertex_index = np.flatnonzero(is_in_area & is_valid)

response_area = response[vertex_index]
area = pd.Categorical([roi_mapping[index] for index in dataset.roi_index[vertex_index]], categories=area_order)

pd.Series(area).value_counts(sort=False).to_frame("num_vertices").T
```

## Step 1: Fitting a pRF model with the Glover HRF

We start with a standard fit: a grid search over the pRF center (`mu_x`, `mu_y`) and size (`sigma`), followed by least squares for `baseline` and `amplitude`, with the Glover HRF of the {py:class}`~prfmodel.impulse.TwoGammaImpulse` model. The model samples the HRF at the frame duration of the stimulus (1.5 seconds).

```{code-cell} ipython3
from prfmodel.fitters import grid_values_2d_prf
from prfmodel.impulse import TwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel

glover_model = Gaussian2DPRFModel(impulse_model=TwoGammaImpulse())

prf_values = grid_values_2d_prf(
    stimulus,
    num_mu=15,
    mu_extent=1.0,  # Centers span the stimulus
    num_sigma=10,
    sigma_range=(0.2, 6.0),  # log-spaced by default
) | {
    "baseline": [0.0],
    "amplitude": [1.0],
}

num_combinations = np.prod([len(values) for values in prf_values.values()])
print(f"Number of parameter combinations: {num_combinations}")
```

We use a relatively coarse grid, because the grid with the HRF library in Step 3 is 20 times larger. In practice, we recommend finer grids.

```{code-cell} ipython3
from prfmodel.fitters import GridFitter, LeastSquaresFitter

glover_grid_history, glover_grid_params = GridFitter(
    model=glover_model,
    stimulus=stimulus,
    compile_step=True,  # Setting 'compile_step=True' speeds up the fitting
).fit(data=response_area, parameter_values=prf_values, batch_size=20)  # Small batches save memory

_, glover_params = LeastSquaresFitter(model=glover_model, stimulus=stimulus).fit(
    data=response_area,
    parameters=glover_grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=1000,
)
glover_params
```

We compute the R-squared of each vertex with {py:func}`prfmodel.utils.calculate_r_squared`.

```{code-cell} ipython3
from prfmodel.utils import calculate_r_squared

glover_response = glover_model(stimulus, glover_params)
glover_r_squared = calculate_r_squared(response_area, glover_response)
```

## Step 2: Selecting vertices that respond to the stimulus

For vertices that do not respond to the stimulus, the selected HRF would only reflect noise. We therefore only select the HRF for vertices that are fitted well by the Glover model and whose pRF center lies inside the stimulus. Note that this selection favors vertices whose response the Glover HRF already describes reasonably well.

```{code-cell} ipython3
r_squared_threshold = 0.3
stimulus_radius = np.abs(stimulus.grid).max()

glover_eccentricity = np.hypot(glover_params["mu_x"], glover_params["mu_y"]).to_numpy()
is_selected = (glover_r_squared > r_squared_threshold) & (glover_eccentricity < stimulus_radius)

response_selected = response_area[is_selected]
area_selected = area[is_selected]

pd.Series(area_selected).value_counts(sort=False).to_frame("num_selected_vertices").T
```

## Step 3: Selecting the HRF with a grid search

We now create a pRF model with an {py:class}`~prfmodel.impulse.HRFLibraryImpulse` as its impulse model and add `"hrf_index": np.arange(20)` to the same pRF grid. The grid search evaluates every combination of pRF parameters with every HRF in the library, so it estimates the pRF and the HRF jointly.

```{code-cell} ipython3
from prfmodel.impulse import HRFLibraryImpulse

hrf_library_model = Gaussian2DPRFModel(impulse_model=HRFLibraryImpulse())

library_values = prf_values | {"hrf_index": np.arange(20)}

num_combinations = np.prod([len(values) for values in library_values.values()])
print(f"Number of parameter combinations: {num_combinations}")
```

```{code-cell} ipython3
library_grid_history, library_grid_params = GridFitter(
    model=hrf_library_model,
    stimulus=stimulus,
    compile_step=True,
).fit(data=response_selected, parameter_values=library_values, batch_size=50)

_, library_params = LeastSquaresFitter(model=hrf_library_model, stimulus=stimulus).fit(
    data=response_selected,
    parameters=library_grid_params,
    slope_name="amplitude",
    intercept_name="baseline",
    batch_size=1000,
)
library_params
```

The selected HRF is stored in the `hrf_index` column, like any other parameter. `hrf_index` is a discrete parameter that indexes the library, so it cannot be estimated with least squares or stochastic gradient descent. The {py:class}`~prfmodel.fitters.LeastSquaresFitter` keeps it at its grid search estimate. When the estimates are finetuned with the {py:class}`~prfmodel.fitters.SGDFitter`, `hrf_index` must be added to the fixed parameters, for example with `fixed_parameters=["hrf_index"]`.

## Comparing the selected HRFs between visual areas

The HRFs in the library are ordered by their time-to-peak, which makes it a convenient summary of the selected HRF. We compute the time-to-peak of each HRF from the library sampled at its native resolution of 0.1 seconds.

```{code-cell} ipython3
library_hrfs = HRFLibraryImpulse(resolution=0.1)(pd.DataFrame({"hrf_index": np.arange(20)}))
library_times = np.asarray(HRFLibraryImpulse(resolution=0.1).get_frames())[0]
time_to_peak = library_times[library_hrfs.argmax(axis=1)]

library_params["time_to_peak"] = time_to_peak[library_params["hrf_index"].astype(int)]
```

We first look at how often each HRF was selected across all selected vertices.

```{code-cell} ipython3
import matplotlib.pyplot as plt

counts = library_params["hrf_index"].astype(int).value_counts().reindex(np.arange(20), fill_value=0)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(counts.index, counts.to_numpy(), color="#3b6ea8", width=0.8)
ax.set(xlabel="Selected hrf_index", ylabel="Number of vertices", xticks=np.arange(20))
ax.spines[["top", "right"]].set_visible(False)
plt.show()
```

Next, we compare the time-to-peak of the selected HRFs between visual areas with {py:func}`prfmodel.plotting.plot_parameter_by_roi`. The error bars show the standard deviation across vertices.

```{code-cell} ipython3
from prfmodel.plotting import plot_parameter_by_roi

present_areas = [name for name in area_order if (area_selected == name).any()]

plot_parameter_by_roi(
    library_params,
    "time_to_peak",
    np.asarray(area_selected),
    order=present_areas,
    xlabel="Visual area",
    ylabel="Time-to-peak of selected HRF (s)",
)
plt.show()
```

```{code-cell} ipython3
library_params.assign(area=area_selected).groupby("area", observed=True)["time_to_peak"].agg(
    ["median", "mean", "std", "count"]
)
```

We also plot the time-to-peak on the flat surface with {py:func}`prfmodel.plotting.plot_surface_stat_map` to see how it is distributed across the cortex. Vertices that were not selected are not shown.

```{code-cell} ipython3
from prfmodel.plotting import plot_surface_stat_map

time_to_peak_surface = np.full(response.shape[0], np.nan)
time_to_peak_surface[vertex_index[is_selected]] = library_params["time_to_peak"].to_numpy()

plot_surface_stat_map(
    mesh,
    time_to_peak_surface,
    vmin=time_to_peak.min(),
    vmax=time_to_peak.max(),
    cmap="Blues",
    title="Time-to-peak of selected HRF (s)",
)
plt.show()
```

## Comparing the fit with the Glover HRF

We compare the R-squared of the model with the selected HRF with that of the Glover model on the same vertices with {py:func}`prfmodel.plotting.plot_r_squared_comparison`.

```{code-cell} ipython3
from prfmodel.plotting import plot_r_squared_comparison

library_r_squared = calculate_r_squared(response_selected, hrf_library_model(stimulus, library_params))

plot_r_squared_comparison(
    glover_r_squared[is_selected],
    library_r_squared,
    value_range=(0.0, 1.0),
    xlabel="R-squared (Glover HRF)",
    ylabel="R-squared (selected HRF)",
    figsize=(5, 5),
)
plt.show()
```

```{code-cell} ipython3
pd.DataFrame(
    {
        "area": area_selected,
        "r_squared_glover": glover_r_squared[is_selected],
        "r_squared_selected_hrf": library_r_squared,
    },
).groupby("area", observed=True).median()
```

Note that both R-squared values are computed on the data that the models were fitted to. Because the model with the HRF library can choose from 20 HRFs, it is expected to fit at least as well as the Glover model even if the hemodynamic response did not differ between vertices. To test whether the selected HRF generalizes, the models should be evaluated on independent data, for example on a separate run with the same stimulus.

+++

## Conclusion

This example showed how to select the best fitting HRF for each vertex from the library of 20 canonical HRFs from GLMsingle. We first fitted a pRF model with the Glover HRF and selected vertices that respond to the stimulus. For these vertices, we then estimated the pRF and the HRF jointly with a grid search over the pRF parameters and `hrf_index`. Finally, we compared the selected HRFs between visual areas and compared the fit with that of the Glover model.

+++

## References

+++

Benson, N. C., Butt, O. H., Brainard, D. H., & Aguirre, G. K. (2014). Correction of distortion in flattened representations of the cortical surface allows prediction of V1-V3 functional organization from anatomy. *PLoS Computational Biology, 10*(3), e1003538. https://doi.org/10.1371/journal.pcbi.1003538

Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife, 11*, e77599. https://doi.org/10.7554/eLife.77599

Roelofzen, C., Daghlian, M., van Dijk, J. A., de Jong, M. C., & Dumoulin, S. O. (2025). Modeling neural contrast sensitivity functions in human visual cortex. *Imaging Neuroscience, 3*, imag_a_00469. https://doi.org/10.1162/imag_a_00469
