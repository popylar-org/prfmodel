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
mystnb:
  output_stderr: remove
---

# Introduction

prfmodel is a modern Python implementation for fitting population receptive field (pRF) and adjacent models,
such as connective field and contrast sensitivity models. It leverages GPU-accelerated backends, such as TensorFlow,
to fit these models at break-neck speed.

```{code-cell} ipython3
:tags: [remove-input]

# Shared setup for the figures on this page: palette, plotly helpers, and the example stimulus.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch
from plotly.subplots import make_subplots
from prfmodel.examples import load_2d_prf_bar_stimulus
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.impulse import convolve_prf_impulse_response
from prfmodel.models.prf import Gaussian2DPRFTuning
from prfmodel.models.prf import encode_prf_response
from prfmodel.models.prf import predict_gaussian_response

%config InlineBackend.figure_formats = ["svg"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
LINE = "#dcdbd6"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

BLUES = [[0.0, SURFACE], [0.2, "#cde2fb"], [0.45, "#86b6ef"], [0.7, "#2a78d6"], [1.0, "#104281"]]
GRAYS = [[0.0, SURFACE], [1.0, "#52514e"]]

plt.rcParams.update(
    {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9, "text.color": INK},
)

LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK),
    margin=dict(l=60, r=20, t=50, b=115),
    hovermode="closest",
    height=430,
)
AXIS = dict(
    showgrid=True,
    gridcolor=LINE,
    zeroline=False,
    linecolor=LINE,
    ticks="outside",
    tickcolor=LINE,
    tickfont=dict(color=MUTED, size=11),
    title_font=dict(size=12, color=MUTED),
)


def show(fig):
    """Render a plotly figure as self-contained HTML that Sphinx can embed."""
    return HTML(
        fig.to_html(
            include_plotlyjs="cdn",
            full_html=False,
            default_width="100%",
            auto_play=False,
            config={"displayModeBar": False, "responsive": True},
        ),
    )


def slider(steps, prefix, **kwargs):
    """Build a slider with the shared styling."""
    return dict(
        steps=steps,
        currentvalue=dict(prefix=prefix, font=dict(size=12, color=MUTED)),
        pad=dict(t=55, b=10),
        font=dict(size=11, color=MUTED),
        bgcolor=LINE,
        bordercolor=LINE,
        activebgcolor=BLUE,
        tickcolor=LINE,
        **kwargs,
    )


def animate_step(name, label):
    """Build a slider step that jumps to the frame `name`."""
    return dict(
        method="animate",
        label=label,
        args=[
            [name],
            dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
        ],
    )


stimulus = load_2d_prf_bar_stimulus()
# The design columns run from positive to negative x, so flip them to plot x from left to right
design = stimulus.design[:, :, ::-1]
grid = stimulus.grid[:, ::-1]
# Plot the visual field on a coarser grid to keep the figures small
grid_coarse = grid[::3, ::3]
x_coords = grid_coarse[0, :, 1]
y_coords = grid_coarse[:, 0, 0]
```

## Design philosophy

prfmodel has been designed with a set of aims in mind:

- It should be **accessible**, so users with less familiarity with the underlying models and fitting methods can use the
package
- It should be **extendible**, so that experienced users can easily build their own models and workflows on top of the
package
- It should follow **best practices**, both regarding the quality of the software and the design
- It should be **fast**, so that users can quickly iterate through their modelling workflow and experiment with the package

To fulfill these aims, we have followed certain design principles:

- **User-friendly public interfaces**: The package implements a public-facade architecture that separates a user-friendly
interface that is compatible standard Python packages such as numpy and pandas from a backend that runs heavy
computations using tensors on GPU-accelerated backends
- **Modularity**: By decomposing models into exchangeable building blocks (submodels), prfmodel facilitates building
customized models that can be fed into customized fitting algorithms through standardized interfaces and contracts
- **Software quality**: The package adheres to research software development best practices (e.g., linting, testing,
continuous integration), tries to provide good defaults, and documents important design decisions
- **Flexible backends**: prfmodel is built on top of [Keras 3.0](https://keras.io/api/), enabling GPU-accelerated
computations through three different backends (TensorFlow, PyTorch, and JAX)

## Models

prfmodel implements many different models that predict a timeseries of measured neural response (e.g., BOLD in fMRI)
towards an experimental stimulus conditional on a set of model parameters. Models can be seen as computational
graphs (typically acyclic) that are composed by different submodels and operations (e.g., convolution). The exact
composition depends on the model family and the experimental stimulus. prfmodel covers many model families
(see overview TBD).

```{code-cell} ipython3
:tags: [remove-input]


def node(ax, x, y, w, h, title, sub, edge):
    """Draw a labelled box of the computational graph."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1,rounding_size=0.12",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=SURFACE,
        ),
    )
    ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=11, color=INK)
    ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center", fontsize=9, color=MUTED, style="italic")


def edge(ax, start, end, style="-", connect="arc3,rad=0.0"):
    """Draw an arrow between two boxes of the computational graph."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=MUTED,
            linestyle=style,
            shrinkA=3,
            shrinkB=3,
            connectionstyle=connect,
        ),
    )


fig, ax = plt.subplots(figsize=(11, 3.9))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3.9)
ax.axis("off")

node(ax, 0.2, 2.75, 2.4, 0.8, "Stimulus", "design $S(t, x, y)$ and grid", MUTED)
node(ax, 0.2, 1.6, 2.4, 0.8, "Tuning profile", r"$g(x, y \mid \mu, \sigma)$", BLUE)
node(ax, 0.2, 0.45, 2.4, 0.8, "Impulse response", r"$h(t \mid delay, \dots)$", BLUE)
node(ax, 3.2, 1.6, 1.7, 0.8, "Encode", r"$\sum_{x,y}$", BLUE)
node(ax, 5.2, 1.6, 1.7, 0.8, "Convolve", r"$\ast$ over time", BLUE)
node(ax, 7.2, 1.6, 1.7, 0.8, "Scale", "amplitude, baseline", BLUE)
node(ax, 9.2, 1.6, 1.7, 0.8, "Predicted response", r"$\hat{y}(t)$", ORANGE)

edge(ax, (2.6, 3.15), (3.2, 2.28))
edge(ax, (1.4, 2.75), (1.4, 2.4))
edge(ax, (2.6, 2.0), (3.2, 2.0))
edge(ax, (2.6, 0.85), (6.05, 1.6), connect="angle,angleA=0,angleB=90,rad=0")
edge(ax, (4.9, 2.0), (5.2, 2.0))
edge(ax, (6.9, 2.0), (7.2, 2.0))
edge(ax, (8.9, 2.0), (9.2, 2.0))

fig.tight_layout()
plt.show()
```

**Figure 1.** The canonical pRF model as a computational graph. The stimulus design and the parameterized submodels
(blue) enter a chain of operations that ends in a predicted response.

### Population tuning

The core set of submodels describe the tuning profile of neuron populations (e.g., all neurons located
within a voxel in fMRI) towards experimental stimuli (e.g., a bar passing through the visual field). For example,
for visual pRF models, the tuning profile is also called the receptive field that is the "region in visual space that
stimulates the recording site" (p. 647)[^dumoulin2008][^victor1994]. For connective field models,
the tuning profile is the stimulation by neuron populations in a source area (e.g., V1)[^haak2013].

Tuning profiles are defined on the feature space of the experimental stimulus. For visual pRF models, the feature space
is typically the 2D visual field (measured in degrees of visual angle). However, for other modalities, the feature
space can be different: For example, numerosity (e.g., the number of objects seen by the subject)[^harvey2013]
or auditory stimuli
are typically defined in 1D logarithmic feature spaces (measured in log integers or log frequency). Connective field
models are always defined on the 2D geodesic distance matrix of the source region.

More advances tuning profiles also account for static nonlinearities, such as surround-suppression (difference of
Gaussian)[^zuiderbaan2012] or saturation (compressive spatial summation)[^kay2013] or both (divisive
normalization)[^aqil2021].

```{code-cell} ipython3
:tags: [remove-input]

sigmas = np.round(np.linspace(0.4, 2.4, 9), 2)
log_numerosity = np.linspace(0.0, 3.4, 240)[:, None]
mu_2d = np.array([[1.45, -2.1]])  # (mu_y, mu_x) in degrees of visual angle
mu_1d = np.array([[np.log(5.0)]])  # centred on a numerosity of five

# The 1D tuning widths are scaled to the (much smaller) range of the log numerosity feature space
def normalized(response):
    """Scale a tuning profile to its peak so that the slider compares shapes, not peak heights."""
    response = np.asarray(response)
    return np.round(response / response.max(), 3)


profiles_2d = [normalized(predict_gaussian_response(grid_coarse, mu_2d, np.array([[s]]))[0]) for s in sigmas]
profiles_1d = [normalized(predict_gaussian_response(log_numerosity, mu_1d, np.array([[s / 6]]))[0]) for s in sigmas]
start = 3

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.13,
    subplot_titles=("Visual field (2D, degrees of visual angle)", "Numerosity (1D, log units)"),
)
fig.add_trace(
    go.Heatmap(
        z=profiles_2d[start],
        x=x_coords,
        y=y_coords,
        colorscale=BLUES,
        zmin=0.0,
        zmax=1.0,
        showscale=False,
        hovertemplate="x %{x:.1f}<br>y %{y:.1f}<br>tuning %{z:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=log_numerosity[:, 0],
        y=profiles_1d[start],
        mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x:.2f} log units<br>tuning %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.frames = [
    go.Frame(
        name=f"sigma{i}",
        traces=[0, 1],
        data=[go.Heatmap(z=profiles_2d[i]), go.Scatter(y=profiles_1d[i])],
    )
    for i in range(len(sigmas))
]

ticks = np.array([1, 2, 4, 8, 16, 25])
fig.update_layout(
    **LAYOUT,
    showlegend=False,
    sliders=[
        slider(
            [animate_step(f"sigma{i}", f"{s:.1f}") for i, s in enumerate(sigmas)],
            "tuning width: ",
            active=start,
        ),
    ],
)
fig.update_xaxes(AXIS, title_text="x-coordinate", row=1, col=1)
fig.update_yaxes(AXIS, title_text="y-coordinate", scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(
    AXIS,
    title_text="Numerosity",
    tickmode="array",
    tickvals=np.log(ticks),
    ticktext=[str(t) for t in ticks],
    row=1,
    col=2,
)
fig.update_yaxes(AXIS, title_text="Tuning (peak-normalized)", range=[-0.05, 1.08], row=1, col=2)
show(fig)
```

**Figure 2.** The same Gaussian tuning model on two feature spaces
({py:class}`~prfmodel.models.prf.Gaussian2DPRFTuning` on the left,
{py:class}`~prfmodel.models.prf.Gaussian1DPRFTuning` on the right). Drag the slider to change the tuning width;
each panel shows the width on the scale of its own feature space. For the purpose of this visualization, both profiles
are normalized to their peak. prfmodel normally uses proper densities that are not peak-normalized (see
{doc}`Important details <important_details>`).

### Stimulus encoding

To predict a temporal neural response, population tuning profiles are often encoded by the design of the experimental
stimulus. For example, for visual pRF models, the design describes the activity in the visual field over time (e.g,
the location of a bar). The design is the component that drives the response of a neuron population in relation to the
feature space of the stimulus. Computationally, the encoding differs between model families: For example, visual pRF
models use the dot product to encode the pRF tuning profile with the stimulus design.

Connective field models do not explicitly use the stimulus design for encoding. Instead they encode the connective
field tuning profile with the neural responses in the source region (i.e., the source region is the "stimulus").
However, the source region neural responses are usually elicited through an actual experimental stimulus that is then
implicitly encoded in the connective field response.

```{code-cell} ipython3
:tags: [remove-input]

prf_parameters = pd.DataFrame({"mu_x": [-2.1], "mu_y": [1.45], "sigma": [1.35]})
tuning_profile = Gaussian2DPRFTuning()(stimulus, prf_parameters)
encoded_response = np.asarray(encode_prf_response(tuning_profile, stimulus.design))[0]

design_coarse = np.round(design[:, ::3, ::3], 2)
# The profile is predicted on the original grid, so flip its columns like the design and grid above
tuning_coarse = np.round(tuning_profile[0, :, ::-1][::3, ::3], 3)
time_frames = list(range(0, design.shape[0], 5))
# Start on the frame closest to the first response peak so that the bar crosses the pRF
peak_frame = min(time_frames, key=lambda t: abs(t - int(np.argmax(encoded_response))))
start_frame = time_frames.index(peak_frame)

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    column_widths=[0.4, 0.6],
    subplot_titles=("Stimulus design and pRF", "Stimulus-encoded response"),
)
fig.add_trace(
    go.Heatmap(
        z=design_coarse[peak_frame],
        x=x_coords,
        y=y_coords,
        colorscale=GRAYS,
        zmin=0.0,
        zmax=1.0,
        showscale=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Contour(
        z=tuning_coarse,
        x=x_coords,
        y=y_coords,
        showscale=False,
        contours_coloring="lines",
        colorscale=[[0.0, ORANGE], [1.0, ORANGE]],
        line=dict(width=2),
        ncontours=5,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=np.arange(encoded_response.size),
        y=np.round(encoded_response, 3),
        mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="frame %{x}<br>response %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=[peak_frame],
        y=[np.round(encoded_response[peak_frame], 3)],
        mode="markers",
        marker=dict(color=ORANGE, size=10, line=dict(color=SURFACE, width=2)),
        hoverinfo="skip",
    ),
    row=1,
    col=2,
)
fig.frames = [
    go.Frame(
        name=f"frame{t}",
        traces=[0, 3],
        data=[go.Heatmap(z=design_coarse[t]), go.Scatter(x=[t], y=[np.round(encoded_response[t], 3)])],
    )
    for t in time_frames
]

fig.update_layout(
    **LAYOUT,
    showlegend=False,
    sliders=[
        slider(
            [animate_step(f"frame{t}", str(t)) for t in time_frames],
            "time frame: ",
            active=start_frame,
        ),
    ],
)
fig.update_xaxes(AXIS, title_text="x-coordinate", row=1, col=1)
fig.update_yaxes(AXIS, title_text="y-coordinate", scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(AXIS, title_text="Time frame", row=1, col=2)
fig.update_yaxes(AXIS, title_text="Response", row=1, col=2)
show(fig)
```

**Figure 3.** Encoding a pRF tuning profile (orange contours) with a moving bar design
({py:func}`~prfmodel.models.prf.encode_prf_response`). Drag the slider through the experiment: the response on the
right peaks whenever the bar overlaps the pRF.

### Impulse response convolution

Depending on how the observed neural responses are measured (e.g., with fMRI), the stimulus-encoded neuron population
responses must be modified to resemble the canonical shape that is inherent to the measurement. For example, BOLD
response measurements in fMRI have a canonical shape following the hemodynamic response function (HRF). To bring
the raw stimulus-encoded neural responses into the measurement space of the observed timecourses, they are usually
convolved with an impulse response. Impulse models predict a response over a short time period that has the canonical
shape of the measurement (e.g., the HRF is typically defined over a duration of 20-30 seconds). The convolved response
then contains the stimulus-encoded neuron population response but also follows the canonical shape of the measurement.
Note that the convolution is linear, an assumption that can be violated under rapid or brief stimulus presentations.
Advanced models, such as delayed normalization[^zhou2019] or compressive spatio-temporal models[^kim2024][^kupers2024],
address these problems.

Connective field models already exhibit the measurement shape in the neural responses of the source region so they
usually do not need the convolution with the impulse response.

```{code-cell} ipython3
:tags: [remove-input]

impulse_model = DerivativeTwoGammaImpulse(duration=32.0, resolution=1.0)
weight_deriv = np.round(np.arange(-3.0, 1.75, 0.25), 1)
impulse_times = np.asarray(impulse_model.get_frames())[0]
impulse_responses = [impulse_model(pd.DataFrame({"weight_deriv": [d]}))[0] for d in weight_deriv]
scale = encoded_response.max()
convolved = [
    np.round(np.asarray(convolve_prf_impulse_response(encoded_response[None, :], h[None, :]))[0] / scale, 3)
    for h in impulse_responses
]
default = int(np.flatnonzero(weight_deriv == -0.5)[0])
impulse_range = [min(h.min() for h in impulse_responses) * 1.2, max(h.max() for h in impulse_responses) * 1.15]
response_range = [
    min(encoded_response.min() / scale, min(c.min() for c in convolved)) * 1.2,
    max(encoded_response.max() / scale, max(c.max() for c in convolved)) * 1.15,
]

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    column_widths=[0.35, 0.65],
    subplot_titles=("Impulse response", "Encoded and convolved response"),
)
fig.add_trace(
    go.Scatter(
        x=impulse_times,
        y=np.round(impulse_responses[default], 4),
        mode="lines",
        line=dict(color=AQUA, width=2),
        showlegend=False,
        hovertemplate="%{x:.1f} s<br>%{y:.4f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=np.arange(encoded_response.size),
        y=np.round(encoded_response / scale, 3),
        mode="lines",
        line=dict(color=BLUE, width=2, dash="dot"),
        name="encoded",
        hovertemplate="frame %{x}<br>%{y:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(
        x=np.arange(encoded_response.size),
        y=convolved[default],
        mode="lines",
        line=dict(color=ORANGE, width=2),
        name="convolved",
        hovertemplate="frame %{x}<br>%{y:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.frames = [
    go.Frame(
        name=f"weight_deriv{d}",
        traces=[0, 2],
        data=[go.Scatter(y=np.round(impulse_responses[i], 4)), go.Scatter(y=convolved[i])],
    )
    for i, d in enumerate(weight_deriv)
]

fig.update_layout(
    **{**LAYOUT, "margin": dict(l=60, r=20, t=90, b=115), "height": 470},
    sliders=[
        slider(
            [animate_step(f"weight_deriv{d}", f"{d:.1f}") for d in weight_deriv],
            "weight_deriv: ",
            active=default,
        ),
    ],
    legend=dict(orientation="h", yanchor="bottom", y=1.13, xanchor="center", x=0.5, font=dict(size=11)),
)
fig.update_xaxes(AXIS, title_text="Time (s)", row=1, col=1)
fig.update_yaxes(AXIS, title_text="Impulse amplitude", range=impulse_range, row=1, col=1)
fig.update_xaxes(AXIS, title_text="Time frame", row=1, col=2)
fig.update_yaxes(AXIS, title_text="Response", range=response_range, row=1, col=2)
show(fig)
```

**Figure 4.** Convolving the encoded response with a derivative two-gamma impulse response
({py:class}`~prfmodel.impulse.DerivativeTwoGammaImpulse`). Drag the slider to move the weight of the derivative of the
impulse response: the convolved response (orange) is a time-shifted and smoothed version of the encoded response
(blue).

### Scaling

To bridge differences in baseline and scaling between predicted model responses and observed timecourses, model
predictions are typically scaled and/or shifted. For some models, amplitude and baseline parameters that define the
scaling and offset are treated as nuisance parameters. In other models (e.g., difference of Gaussian and
divisive normalization models), amplitudes are treated as interpretable parameters and only baseline parameters as
nuisances.

```{code-cell} ipython3
:tags: [remove-input]

response = convolved[default]
frame_index = np.arange(response.size)
amplitudes = np.round(np.linspace(0.1, 2.0, 13), 2)
baselines = np.round(np.linspace(-2.0, 2.0, 9), 2)
amplitude_start = int(np.argmin(np.abs(amplitudes - 1.0)))
baseline_start = int(np.argmin(np.abs(baselines)))

fig = make_subplots(
    rows=1,
    cols=2,
    horizontal_spacing=0.12,
    subplot_titles=("Amplitude scales the response", "Baseline shifts the response"),
)
fig.add_trace(
    go.Scatter(
        x=frame_index,
        y=np.round(amplitudes[amplitude_start] * response, 3),
        mode="lines",
        line=dict(color=BLUE, width=2),
        hovertemplate="frame %{x}<br>%{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=frame_index, y=response, mode="lines", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=frame_index,
        y=np.round(response + baselines[baseline_start], 3),
        mode="lines",
        line=dict(color=ORANGE, width=2),
        hovertemplate="frame %{x}<br>%{y:.2f}<extra></extra>",
    ),
    row=1,
    col=2,
)
fig.add_trace(
    go.Scatter(x=frame_index, y=response, mode="lines", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"),
    row=1,
    col=2,
)

amplitude_steps = [
    dict(method="restyle", label=f"{a:.1f}", args=[{"y": [list(np.round(a * response, 3))]}, [0]]) for a in amplitudes
]
baseline_steps = [
    dict(method="restyle", label=f"{b:.1f}", args=[{"y": [list(np.round(response + b, 3))]}, [2]]) for b in baselines
]

fig.update_layout(
    **LAYOUT,
    showlegend=False,
    sliders=[
        slider(amplitude_steps, "amplitude: ", active=amplitude_start, x=0.0, len=0.42),
        slider(baseline_steps, "baseline: ", active=baseline_start, x=0.58, len=0.42),
    ],
)
fig.update_xaxes(AXIS, title_text="Time frame", row=1, col=1)
fig.update_yaxes(AXIS, title_text="Predicted response", range=[-3.0, 3.6], row=1, col=1)
fig.update_xaxes(AXIS, title_text="Time frame", row=1, col=2)
fig.update_yaxes(AXIS, title_text="Predicted response", range=[-3.0, 3.6], row=1, col=2)
show(fig)
```

**Figure 5.** Scaling the convolved response with {py:class}`~prfmodel.scaling.BaselineAmplitude`. The dotted line is
the unscaled response.

## Fitting

In prfmodel, a model that makes predictions can be compared to observed neural responses to optimize the model parameters
and minimize the discrepancy between the two. The package provides several fitting methods that are best applied in
sequence (this part has been inspired by the [braincoder](https://github.com/Gilles86/braincoder) Python package).

The standard fitting workflow starts with a grid search over the core parameters of the model. For example, in the
visual Gaussian pRF model, the core parameters are the center of the Gaussian (i.e., the mean) and the size of the
receptive field (i.e., the standard deviation). By casting a wide grid, the search ensures that the core model
parameters estimates fall in the region that provides predictions that are close to the observed data. The grid search
is followed by least-squares estimation of baseline and amplitude parameters. Finally, some or all model parameters are
finetuned with stochastic gradient descent (SGD) using the estimates from the previous stages as starting values. SGD
is efficient at scale and benefits from the GPU-accelerated backends.

Because model fitting is fast, users can quickly iterate through the fitting stages and move from simple models to
incrementally more complex ones. The {doc}`Getting started <getting_started>` page walks through this workflow with
{py:class}`~prfmodel.fitters.GridFitter`, {py:class}`~prfmodel.fitters.LeastSquaresFitter`, and
{py:class}`~prfmodel.fitters.SGDFitter`.

## Diagnostics

After models are fit to observed data and parameters are estimated, prfmodel helps the user to assess the quality of
the fit and parameter estimates.

## What prfmodel is not

prfmodel is not a package for preprocessing neuroimaging data since this is already extensively covered by other
software and would massively blow up the scope of the package. For the same reasons, it is also not designed to be
a package for visualizing neuroimaging data. We instead rely on [nilearn](https://nilearn.github.io/stable/index.html)
for this task, but also suggest that MacOS and Linux users experiment with [pycortex](https://gallantlab.org/pycortex/)
for visualization.

## References

[^aqil2021]: Aqil, M., Knapen, T., & Dumoulin, S. O. (2021). Divisive normalization unifies disparate response
    signatures throughout the human visual hierarchy. *Proceedings of the National Academy of Sciences*, *118*(46),
    e2108713118. [https://doi.org/10.1073/pnas.2108713118](https://doi.org/10.1073/pnas.2108713118)

[^dumoulin2008]: Dumoulin, S. O., & Wandell, B. A. (2008). Population receptive field estimates in human visual cortex.
    *NeuroImage*, *39*(2), 647–660.
    [https://doi.org/10.1016/j.neuroimage.2007.09.034](https://doi.org/10.1016/j.neuroimage.2007.09.034)

[^haak2013]: Haak, K. V., Winawer, J., Harvey, B. M., Renken, R., Dumoulin, S. O., Wandell, B. A., & Cornelissen, F. W.
    (2013). Connective field modeling. *NeuroImage*, *66*, 376–384.
    [https://doi.org/10.1016/j.neuroimage.2012.10.037](https://doi.org/10.1016/j.neuroimage.2012.10.037)

[^harvey2013]: Harvey, B. M., Klein, B. P., Petridou, N., & Dumoulin, S. O. (2013). Topographic representation of
    numerosity in the human parietal cortex. *Science*, *341*(6150), 1123–1126.
    [https://doi.org/10.1126/science.1239052](https://doi.org/10.1126/science.1239052)

[^kay2013]: Kay, K. N., Winawer, J., Mezer, A., & Wandell, B. A. (2013). Compressive spatial summation in human visual
    cortex. *Journal of Neurophysiology*, *110*(2), 481–494.
    [https://doi.org/10.1152/jn.00105.2013](https://doi.org/10.1152/jn.00105.2013)

[^kim2024]: Kim, I., Kupers, E. R., Lerma-Usabiaga, G., & Grill-Spector, K. (2024). Characterizing spatiotemporal
    population receptive fields in human visual cortex with fMRI. *The Journal of Neuroscience*, *44*(2), e0803232023.
    [https://doi.org/10.1523/JNEUROSCI.0803-23.2023](https://doi.org/10.1523/JNEUROSCI.0803-23.2023)

[^kupers2024]: Kupers, E. R., Kim, I., & Grill-Spector, K. (2024). Rethinking simultaneous suppression in visual cortex
    via compressive spatiotemporal population receptive fields. *Nature Communications*, *15*(1), 6885.
    [https://doi.org/10.1038/s41467-024-51243-7](https://doi.org/10.1038/s41467-024-51243-7)

[^victor1994]: Victor, J. D., Purpura, K., Katz, E., & Mao, B. (1994). Population encoding of spatial frequency,
    orientation, and color in macaque V1. *Journal of Neurophysiology*, *72*(5), 2151–2166.
    [https://doi.org/10.1152/jn.1994.72.5.2151](https://doi.org/10.1152/jn.1994.72.5.2151)

[^zhou2019]: Zhou, J., Benson, N. C., Kay, K., & Winawer, J. (2019). Predicting neuronal dynamics with a delayed gain
    control model. *PLOS Computational Biology*, *15*(11), e1007484.
    [https://doi.org/10.1371/journal.pcbi.1007484](https://doi.org/10.1371/journal.pcbi.1007484)

[^zuiderbaan2012]: Zuiderbaan, W., Harvey, B. M., & Dumoulin, S. O. (2012). Modeling center–surround configurations in
    population receptive fields using fMRI. *Journal of Vision*, *12*(3), 10.
    [https://doi.org/10.1167/12.3.10](https://doi.org/10.1167/12.3.10)
