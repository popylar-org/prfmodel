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

# How the hemodynamic response function shapes the predicted response

+++

**Author**: Blazej M. Baczkowski (b.baczkowski@esciencecenter.nl)

**Difficulty**: Beginner

+++

This tutorial introduces the library of canonical hemodynamic response functions (HRFs) in prfmodel, and shows how the HRF changes the response that a population receptive field (pRF) model predicts.

The shape of the HRF differs between brain regions and between individuals. A pRF model typically uses the same HRF (e.g., the Glover HRF) for all units. When the hemodynamic response of a unit peaks earlier or later than this HRF, the predicted response is shifted in time relative to the observed response, and the model fits that unit poorly. Instead of estimating the HRF parameters freely, we can choose the best fitting HRF from a library of canonical HRFs. prfmodel ships the library of 20 canonical HRFs from GLMsingle (Prince et al., 2022), which spans the range of HRF shapes observed in the Natural Scenes Dataset (Allen et al., 2022). The {py:class}`~prfmodel.impulse.HRFLibraryImpulse` impulse model selects an HRF from this library with its single parameter `hrf_index`, so that the HRF can be searched in the same grid as the pRF parameters.

+++

Because prfmodel uses Keras for model fitting, we need to make sure that a backend is installed before we begin.
In this tutorial, we use the TensorFlow backend.

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

## The canonical HRF library

Let's start by looking at the HRFs in the library. We create an {py:class}`~prfmodel.impulse.HRFLibraryImpulse` model and predict the impulse response of all 20 HRFs. We sample them every 0.1 seconds (the native resolution of the library) and normalize them to a peak of one, so that their shapes are easy to compare.

```{code-cell} ipython3
import numpy as np
from prfmodel.impulse import HRFLibraryImpulse

library_model = HRFLibraryImpulse(resolution=0.1, norm="max")

all_hrfs = pd.DataFrame({"hrf_index": np.arange(20)})
hrfs = library_model(all_hrfs)
times = np.asarray(library_model.get_frames())[0]

print(hrfs.shape)  # (num_hrfs, num_frames)
```

```{code-cell} ipython3
import matplotlib as mpl
import matplotlib.pyplot as plt

# The HRFs are ordered, so a single sequential hue encodes the index; the lightest steps are skipped so that every
# line stays visible against the white background
cmap = mpl.colors.ListedColormap(plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
colors = cmap(np.linspace(0.0, 1.0, len(hrfs)))

fig, ax = plt.subplots(figsize=(7, 4))

for hrf, color in zip(hrfs, colors):
    ax.plot(times, hrf, color=color, linewidth=1.5)

ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
ax.set(xlabel="Time (s)", ylabel="Response (peak = 1)", xlim=(0, 30))
ax.spines[["top", "right"]].set_visible(False)
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 19), cmap=cmap), ax=ax, label="hrf_index")
plt.show()
```

The HRFs are ordered from early to late time-to-peak. HRFs with a low index peak early and have a pronounced undershoot, while HRFs with a high index peak late and return slowly to zero.

```{code-cell} ipython3
time_to_peak = pd.Series(times[hrfs.argmax(axis=1)], index=pd.Index(np.arange(20), name="hrf_index"))
time_to_peak.to_frame("time_to_peak").T
```

## How the HRF changes the predicted response

To see how much the HRF matters, we predict the response of a single pRF to a bar stimulus that moves through a two-dimensional visual field, once with several HRFs from the library and once with the Glover HRF.

```{code-cell} ipython3
from prfmodel.examples import load_2d_prf_bar_stimulus

stimulus = load_2d_prf_bar_stimulus()
print(stimulus)
```

We create two Gaussian 2D pRF models that only differ in their impulse model. The first uses an {py:class}`~prfmodel.impulse.HRFLibraryImpulse`, the second a {py:class}`~prfmodel.impulse.TwoGammaImpulse` with its default parameters, the Glover HRF. Neither model sets the `resolution` of its impulse model, so both sample their HRF at the frame duration of the stimulus (0.9 seconds).

```{code-cell} ipython3
from prfmodel.impulse import TwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel

hrf_library_model = Gaussian2DPRFModel(impulse_model=HRFLibraryImpulse())
glover_model = Gaussian2DPRFModel(impulse_model=TwoGammaImpulse())

print(hrf_library_model.parameter_names)
print(glover_model.parameter_names)
```

Both models have the pRF center (`mu_x`, `mu_y`) and size (`sigma`) and the `baseline` and `amplitude` of the scaling model. The library model has the additional parameter `hrf_index`. The parameters of the Glover HRF are not listed because the {py:class}`~prfmodel.impulse.TwoGammaImpulse` supplies them as default parameters.

We choose one pRF and five HRFs from the library that cover the range from early to late time-to-peak.

```{code-cell} ipython3
prf_params = {"mu_x": 1.6, "mu_y": -0.8, "sigma": 1.0, "baseline": 0.0, "amplitude": 1.0}
hrf_indices = [0, 5, 10, 15, 19]

library_params = pd.DataFrame({**prf_params, "hrf_index": hrf_indices})
glover_params = pd.DataFrame(prf_params, index=[0])

library_response = hrf_library_model(stimulus, library_params)
glover_response = glover_model(stimulus, glover_params)

print(library_response.shape, glover_response.shape)  # (num_units, num_frames)
```

We plot the predicted responses on top of each other.

```{code-cell} ipython3
frame_times = np.arange(library_response.shape[1]) * stimulus.frame_duration

fig, ax = plt.subplots(figsize=(10, 4))

for hrf_index, response in zip(hrf_indices, library_response):
    label = f"hrf_index = {hrf_index} (peak at {time_to_peak[hrf_index]:.1f} s)"
    ax.plot(frame_times, response, color=cmap(hrf_index / 19), linewidth=1.5, label=label)

ax.plot(frame_times, glover_response[0], color="black", linewidth=2.0, linestyle="--", label="Glover HRF")

ax.set(xlabel="Time (s)", ylabel="Predicted response")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize="small")
plt.show()
```

All predictions respond to the same bar sweeps, but the response peaks later and is smoother for HRFs with a higher index.

To quantify the difference, we correlate each prediction with the prediction of the Glover HRF. This is the measure that the default {py:class}`~prfmodel.fitters.losses.CorrelationLoss` of the {py:class}`~prfmodel.fitters.GridFitter` uses (with a flipped sign). A correlation below one means that, for this pRF, the Glover model cannot reproduce the response of a unit with this HRF exactly.

```{code-cell} ipython3
correlation_with_glover = [np.corrcoef(response, glover_response[0])[0, 1] for response in library_response]

pd.DataFrame(
    {
        "time_to_peak": time_to_peak[hrf_indices].to_numpy(),
        "correlation_with_glover": correlation_with_glover,
    },
    index=pd.Index(hrf_indices, name="hrf_index"),
)
```

The correlation is highest for the library HRFs that peak at a similar time as the Glover HRF (about 5 seconds) and lower for HRFs that peak earlier.

+++

## Conclusion

In this tutorial, we looked at the library of 20 canonical HRFs from GLMsingle and saw that the same pRF predicts a noticeably different response depending on the HRF. The example {doc}`Selecting the hemodynamic response function for each vertex in fMRI data from a visual experiment </tutorials/examples/prf_2d_fmri_hrf_selection>` shows how to select the best fitting HRF for each vertex with a grid search and compares the selected HRFs between visual areas.

+++

## References

+++

Allen, E. J., St-Yves, G., Wu, Y., Breedlove, J. L., Prince, J. S., Dowdle, L. T., Nau, M., Caron, B., Pestilli, F., Charest, I., Hutchinson, J. B., Naselaris, T., & Kay, K. (2022). A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial intelligence. *Nature Neuroscience, 25*(1), 116–126. https://doi.org/10.1038/s41593-021-00962-x

Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife, 11*, e77599. https://doi.org/10.7554/eLife.77599
