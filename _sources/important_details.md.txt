# Important details

When developing prfmodel, we made certain decisions for the design and implementation of different features.
While the decisions concerning the development of prfmodel are covered in [Development](development/index.rst) (e.g., architecture),
we describe the choices that affect **users** directly in this section.

## Gaussian models use proper probability densities

Some software packages that implement Gaussian population receptive field (pRF) models use unnormalized Gaussian
densities to predict tuning profiles. That is, they use the density:
\begin{equation}
f(x) = e^{-\frac{\lVert x - \mu \rVert^2}{2 \sigma^2}},
\end{equation}
that has a peak amplitude of $\max(f(x)) = 1$. Although this parameterization has advantages in some situations,
we decided that all Gaussian models in prfmodel use proper densities that are normalized by their
volume (see {py:func}`~prfmodel.density.normal_density`):
\begin{equation}
f(x) = \frac{1}{V} e^{-\frac{\lVert x - \mu \rVert^2}{2 \sigma^2}},
\end{equation}
where $V = (2 \pi \sigma^2)^{k / 2}$ is the volume and $k$ is the number of dimensions of the tuning profile.
The proper density has a peak amplitude of $\max(f(x)) = 1/V$.

The proper Gaussian density has the advantage that it decouples amplitude parameters from pRF size/tuning width
parameters $\sigma$, making them easier to interpret (although amplitudes are often treated as nuisance parameters).
Because the density peak is $1/V$, stimulus-encoded responses are scaled by $1/V$, and amplitude estimates are scaled
by $V$ (i.e., larger amplitudes for $V > 1$). However, this decoupling does **not** change the identifiability of the
model parameters.

Using the proper Gaussian density implies that parameter estimates for amplitudes
cannot be directly compared to the estimates from software that uses the unnormalized density. This holds not only
for the Gaussian 1D and 2D pRF models but also for difference of Gaussian and Gaussian divisive normalization pRF
models as well as Gaussian connective field models.

It is possible to convert the amplitudes estimated with the proper density into those estimated with unnormalized
density by dividing by the volume:
\begin{equation}
\beta_\text{unnorm} = \frac{\beta_\text{norm}}{V}.
\end{equation}
Importantly, this conversion assumes that the models for which the amplitudes have been estimated are otherwise equal.

Note that you can implement your own Gaussian models in prfmodel that use unnormalized densities
(see the tutorial on [custom models](tutorials/tutorials/custom_models.md)).

## Spatial receptive fields are not scaled by grid cell size

Some software packages normalize stimulus-encoded model responses of spatial models by the
size of the cells in the spatial grid. For example, for the Gaussian 2D pRF model in visual space, the stimulus-encoded
response can be normalized as follows:
\begin{equation}
r(t) = \sum_{xy} g(x, y) \cdot S(t, x, y) dA,
\end{equation}
where $g(x, y)$ is the Gaussian pRF tuning profile, $S(t, x, y)$ is the stimulus design, and $dA = dx dy$ is the
size of each cell in the grid. This normalization changes
the scale and the interpretation of amplitude parameters, making them comparable across spatial grid resolutions.

An alternative convention adopted by some software packages is to normalize spatial RFs (called tuning
profiles in prfmodel) by their sum[^1]. This makes the conflict between volume-normalized vs -unnormalized densities
irrelevant and amplitudes comparable across grid cell sizes. However, when a pRF only partially falls within the
stimulus coordinate grid, normalizing it inflates the tuning strength for the overlapping coordinates and biases
amplitude estimates. Therefore, the normalization ties the amplitude of these pRFs to the sizes of the
spatial coordinate dimensions (e.g., width and height)[^2].

Normalization also leads to numerical issues when RFs are not covered by the
stimulus grid because then their sum is zero. Moreover, because some spatial models have irregular-spaced
one-hot-encoded grids (e.g., the Gaussian 1D pRF model in log numerosity space) and non-spatial models
(e.g., the Gaussian connective field model) do not have an equivalent
grid cell size or meaningful normalizations, we decided **against** normalizing spatial models in prfmodel.
We also do not want to tie our model implementations to the spatial domain.

This decision means that amplitude parameters are not comparable across different spatial grid resolutions (e.g.,
upsampling a $128^2$ to a $256^2$ grid while keeping the overall width and height would shrink amplitude estimates
by 4). However, for regular-spaced grids, it is possible to divide amplitudes by the grid cell size to make them
comparable across grid resolutions. Normalizing (or not) does **not** affect the identifiability of model parameters.

## Impulse responses are only normalized when they describe measurements of neural activity

In prfmodel, impulse models can describe both the canonical shape of a measurement of brain activity (e.g., the
BOLD response in fMRI) and the activity of neuron populations. When they describe a measurement of neural activity,
the predicted impulse response is sum-normalized by default (other normalizations can be chosen via the `norm`
argument, see {py:class}`~prfmodel.impulse.DerivativeTwoGammaImpulse`)[^3]. However, when they describe the behavior
of a neuron population, the predicted response is not normalized because its scale is part of the neural dynamics that
the model describes (see e.g., {py:class}`~prfmodel.impulse.SustainedImpulse`).

For the measurement impulse models, the sum-normalization decouples amplitude parameters from impulse response
parameters (e.g., the shape of the gamma distribution), but it does **not** affect the identifiability of model
parameters. With sum-normalization, a sustained stimulus with encoded response $r$ produces a plateau of exactly
$\beta \times r$ (unit DC gain), making the amplitude easier to interpret. It also makes the amplitude approximately
invariant to the resolution of the impulse response (the approximation improves with finer resolution).

It is possible to convert impulse-sum-normalized into impulse-unnormalized amplitudes:
\begin{equation}
\beta_\text{unnorm} = \beta_\text{norm} / \sum_t h_\text{unnorm}(t),
\end{equation}
where $h_\text{unnorm}(t)$ is the unnormalized impulse response.

## Impulse responses must have the same sampling rate as observed responses

Discrete convolution assumes that the convolved signals have the same sampling rate. In prfmodel, the sampling rate
of stimulus designs and observed neural responses is implicit. They are represented as series of time frames that
typically have the same sampling rate (the TR; repetition time), but for some models, they can differ (e.g., in the
compressive spatio-temporal pRF model). For model predictions to be correct, the sampling rate of impulse responses
must match the sampling rate of the stimulus design. By default, the `resolution` (sampling interval) of impulse models
in prfmodel is 1 second, which gives silently wrong predictions when the stimulus design has a different sampling
rate. Therefore, the `resolution` of the impulse model must always be matched to the stimulus design. For a Gaussian 2D
pRF model, this can be done by adding a custom impulse model:

```python
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel


TR = 1.5  # in seconds

# Create a custom impulse model with the TR as resolution
impulse_model = DerivativeTwoGammaImpulse(resolution=TR)

# Insert the custom impulse model into the canonical pRF model
prf_model = Gaussian2DPRFModel(
    impulse_model=impulse_model,
)
```

This implementation might seem a bit cumbersome, however, it forces the user to think explicitly about the sampling
rates used in the model and the experiment. It also becomes helpful as soon as different model components operate on
different sampling rates that must be aligned with each other (e.g., in the compressive spatio-temporal pRF model).
When comparing model predictions against observed timecourses, both must have matching sampling rates too (typically
the TR), otherwise one or the other must be up- or downsampled.

## Impulse responses are sampled at the leading edge of each time frame by default

By default, prfmodel assumes that the TR of observed neural timecourses is **locked to the onset of a stimulus design
frame** (e.g., BOLD measurements in fMRI have been slice-time corrected with fMRIPrep using `--slice-time-ref 0`).
This implies that time frames of observed neural timecourses represent instantaneous measurements of brain activity at
each TR (not averages over an interval). Without slice-time correction, the alignment differs between slices and
an offset (see below) can only approximate it.

Without any up- or downsampling, stimulus design frame $i$, observed sample $i$ and impulse response frame $i$ all
refer to the time $i \cdot \text{TR}$. By default, we therefore sample impulse responses at the **leading edge** of each
frame.

With the default `offset` of zero, the first sample of the kernel is at $t=0$, and the default impulse model
{py:class}`~prfmodel.impulse.DerivativeTwoGammaImpulse` returns exactly zero there. Discrete convolution in prfmodel
treats the first frame of the impulse response as lag 0, so an impulse response of zero means that a stimulus cannot
contribute to the observed neural response during its exact onset (which is biologically plausible).

The leading-edge sampling assumption can be changed by setting an offset in the impulse model. For example, for
mid-frame sampling (i.e., response measurements align with the *center* of a stimulus design frame, mirroring the
default slice-time correction reference of 0.5 in fMRIPrep), the offset should be specified as `TR/2.0`:

```python
from prfmodel.impulse import DerivativeTwoGammaImpulse
from prfmodel.models.prf import Gaussian2DPRFModel


TR = 1.5  # in seconds

# Create a custom impulse model with the TR as resolution and a positive offset
impulse_model = DerivativeTwoGammaImpulse(resolution=TR, offset=TR / 2.0)

# Insert the custom impulse model into the canonical pRF model
prf_model = Gaussian2DPRFModel(
    impulse_model=impulse_model,
)
```

## Before convolution, stimulus-encoded model responses are padded with their first frame

To make sure that convolving the stimulus-encoded model response $r(t)$ with the impulse response $h(t)$ returns model
predictions for the same number of time frames as the stimulus design, we pad stimulus-encoded model responses with
their first frame. Specifically, we first prepend `len(h) - 1` copies of the first stimulus-encoded response frame and
then convolve both signals using discrete convolution.

This choice rests on the assumption that the first stimulus design frame was present indefinitely before the observed
timecourse and, thus, the model prediction at $t = 0$ is the sustained response to the first frame. If the stimulus
design starts at baseline (e.g., a blank screen in 2D visual space) for at least the duration of the impulse response,
this is the resting-state response. If the design starts with a stimulus, predictions for the first frames assume
that this stimulus was already shown for an extended time, so consider starting runs with a blank period. Stimuli from
previous runs in an experiment should not influence the recording of the response to the current stimulus.

## What if I want to deviate from these decisions?

If you have good reasons to deviate from our decisions, you can implement your own models in prfmodel that use
different conventions (see the tutorial on [custom models](tutorials/tutorials/custom_models.md)).

[^1]: Alternative spatial normalization functions (e.g., L2 norm) are possible but bring similar and sometimes even
more problems.

[^2]: It is also possible to normalize the stimulus-encoded response which couples amplitudes to the stimulus
design instead of the grid. This comes with similar problems as normalizing the RF.

[^3]: Normalizing by the L2 norm is actually numerically more stable (because it is never zero for signed impulse
responses) but it also couples amplitudes to impulse model parameters because it does not preserve unit DC gain.
